import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="ECOMM Network Traffic Classifier",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .prediction-normal {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .prediction-malicious {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all pre-trained models and scalers"""
    try:
        scaler = joblib.load('minmaxscaler.pkl')
        pca = joblib.load('pcamodel.pkl')
        
        with open('LIME.pkl', 'rb') as model_file:
            rf_model = pickle.load(model_file)

        model = Sequential()
        model.add(Conv1D(64, 3, padding="same", input_shape=(None, 1), activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Conv1D(64, 3, padding="same", activation='relu', kernel_regularizer=l2(0.01)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(128, 3, padding="same", activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Conv1D(128, 3, padding="same", activation='relu', kernel_regularizer=l2(0.001)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(LSTM(units=100, return_sequences=False, dropout=0.1, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(units=2, activation='softmax', kernel_regularizer=l2(0.01)))
        
        opt = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.load_weights('model.h5')
        
        return scaler, pca, rf_model, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def create_feature_input_form():
    
    st.subheader(" Network Traffic Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Basic Flow Information**")
        dest_port = st.number_input("Destination Port", value=88, min_value=0, max_value=65535)
        flow_duration = st.number_input("Flow Duration (microseconds)", value=640, min_value=0)
        total_fwd_packets = st.number_input("Total Forward Packets", value=7, min_value=0)
        total_bwd_packets = st.number_input("Total Backward Packets", value=4, min_value=0)
        
        st.markdown("**Packet Length Statistics**")
        fwd_pkt_len_max = st.number_input("Forward Packet Length Max", value=220, min_value=0)
        fwd_pkt_len_min = st.number_input("Forward Packet Length Min", value=0, min_value=0)
        fwd_pkt_len_mean = st.number_input("Forward Packet Length Mean", value=62.86, min_value=0.0)
        fwd_pkt_len_std = st.number_input("Forward Packet Length Std", value=107.35, min_value=0.0)
    
    with col2:
        st.markdown("**Backward Packet Information**")
        bwd_pkt_len_max = st.number_input("Backward Packet Length Max", value=179, min_value=0)
        bwd_pkt_len_min = st.number_input("Backward Packet Length Min", value=0, min_value=0)
        bwd_pkt_len_mean = st.number_input("Backward Packet Length Mean", value=89.5, min_value=0.0)
        bwd_pkt_len_std = st.number_input("Backward Packet Length Std", value=103.35, min_value=0.0)
        
        st.markdown("**Flow Statistics**")
        flow_bytes_per_sec = st.number_input("Flow Bytes/s", value=1246875.0, min_value=0.0)
        flow_packets_per_sec = st.number_input("Flow Packets/s", value=17187.5, min_value=0.0)
        flow_iat_mean = st.number_input("Flow IAT Mean", value=64.0, min_value=0.0)
        flow_iat_std = st.number_input("Flow IAT Std", value=135.56, min_value=0.0)
    
    with col3:
        st.markdown("**Advanced Features**")
        total_len_fwd = st.number_input("Total Length Forward Packets", value=440, min_value=0)
        total_len_bwd = st.number_input("Total Length Backward Packets", value=358, min_value=0)
        fwd_header_len = st.number_input("Forward Header Length", value=164, min_value=0)
        bwd_header_len = st.number_input("Backward Header Length", value=104, min_value=0)
        
        st.markdown("**TCP Flags**")
        fin_flag_count = st.number_input("FIN Flag Count", value=0, min_value=0)
        syn_flag_count = st.number_input("SYN Flag Count", value=0, min_value=0)
        rst_flag_count = st.number_input("RST Flag Count", value=0, min_value=0)
        psh_flag_count = st.number_input("PSH Flag Count", value=1, min_value=0)
    
 
    features = [
        dest_port, flow_duration, total_fwd_packets, total_bwd_packets, total_len_fwd, total_len_bwd,
        fwd_pkt_len_max, fwd_pkt_len_min, fwd_pkt_len_mean, fwd_pkt_len_std,
        bwd_pkt_len_max, bwd_pkt_len_min, bwd_pkt_len_mean, bwd_pkt_len_std,
        flow_bytes_per_sec, flow_packets_per_sec, flow_iat_mean, flow_iat_std, 445, 1,
        640, 106.67, 194.33, 497, 1, 538, 179.33, 303.69, 530, 4,
        0, 0, 0, 0, fwd_header_len, bwd_header_len, 10937.5, 6250, 0, 220, 66.5,
        99.00, 9801.36, fin_flag_count, syn_flag_count, rst_flag_count, psh_flag_count, 0, 0, 0, 0, 0,
        72.55, 62.86, 89.5, 164, 0, 0, 0, 0, 0, 0, 7, 440, 4, 358,
        8192, 2053, 2, 20, 0, 0, 0, 0, 0, 0, 0, 0
    ]
    
    return features

def predict_traffic(features, scaler, pca, rf_model, cnn_lstm_model):
    """Make prediction on network traffic"""
    try:
    
        feature_names = [
            'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
            'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
            'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate','Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        
        X_single = pd.DataFrame([features], columns=feature_names[:len(features)])
        
     
        X_single.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_single.fillna(X_single.mean(), inplace=True)
        
     
        X_scaled = scaler.transform(X_single)
        X_pca = pca.transform(X_scaled)
        
    
        rf_prob = rf_model.predict_proba(X_pca)
        
     
        X_reshaped = np.reshape(X_pca, (X_pca.shape[0], X_pca.shape[1], 1))
        cnn_prob = cnn_lstm_model.predict(X_reshaped)
        
       
        fused_probs = (cnn_prob + rf_prob) / 2
        prediction = np.argmax(fused_probs, axis=1)[0]
        confidence = np.max(cnn_prob, axis=1)[0]
        
        return prediction, confidence, fused_probs[0], rf_prob[0], cnn_prob[0]
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None, None

def create_prediction_visualization(rf_prob, cnn_prob, fused_prob):
    """Create visualization of model predictions"""
    
  
    fig_models = go.Figure()
    
    models = ['Random Forest', 'CNN-LSTM', 'Ensemble']
    normal_probs = [rf_prob[0], cnn_prob[0], fused_prob[0]]
    malicious_probs = [rf_prob[1], cnn_prob[1], fused_prob[1]]
    
    fig_models.add_trace(go.Bar(
        name='Normal Traffic',
        x=models,
        y=normal_probs,
        marker_color='#667eea',
        text=[f'{p:.1%}' for p in normal_probs],
        textposition='auto'
    ))
    
    fig_models.add_trace(go.Bar(
        name='Malicious Traffic',
        x=models,
        y=malicious_probs,
        marker_color='#f5576c',
        text=[f'{p:.1%}' for p in malicious_probs],
        textposition='auto'
    ))
    
    fig_models.update_layout(
        title='Model Predictions Comparison',
        yaxis_title='Probability',
        barmode='group',
        height=400,
        showlegend=True
    )
    
    return fig_models
def process_batch_predictions(df, scaler, pca, rf_model, cnn_lstm_model):
   
    try:
     
        X_batch = df.copy()
        
     
        X_batch.replace([np.inf, -np.inf], np.nan, inplace=True)
        
       
        for col in X_batch.columns:
            if X_batch[col].dtype in ['float64', 'int64']:
                X_batch[col].fillna(X_batch[col].mean(), inplace=True)
        
        
        expected_features = len(scaler.feature_names_in_)
        if len(X_batch.columns) < expected_features:
           
            for i in range(len(X_batch.columns), expected_features):
                X_batch[f'feature_{i}'] = 0
        elif len(X_batch.columns) > expected_features:
            
            X_batch = X_batch.iloc[:, :expected_features]
        
        
        X_batch.columns = scaler.feature_names_in_
        
       
        X_scaled = scaler.transform(X_batch)
        X_pca = pca.transform(X_scaled)
        
       
        rf_probs = rf_model.predict_proba(X_pca)
        
      
        X_reshaped = np.reshape(X_pca, (X_pca.shape[0], X_pca.shape[1], 1))
        cnn_probs = cnn_lstm_model.predict(X_reshaped)
        
        
        fused_probs = (cnn_probs + rf_probs) / 2
        predictions = np.argmax(fused_probs, axis=1)
        confidences = np.max(fused_probs, axis=1)
        
        return predictions, confidences, fused_probs
    
    except Exception as e:
        st.error(f"Batch processing error: {str(e)}")
        return None


    """Make prediction on network traffic"""
    try:
       
        feature_names = [
            'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
            'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
            'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
            'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate','Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        
        X_single = pd.DataFrame([features], columns=feature_names[:len(features)])
        
     
        X_single.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_single.fillna(X_single.mean(), inplace=True)
        
        
        X_scaled = scaler.transform(X_single)
        X_pca = pca.transform(X_scaled)
        
       
        rf_prob = rf_model.predict_proba(X_pca)
        
      
        X_reshaped = np.reshape(X_pca, (X_pca.shape[0], X_pca.shape[1], 1))
        cnn_prob = cnn_lstm_model.predict(X_reshaped)
        
      
        fused_probs = (cnn_prob + rf_prob) / 2
        prediction = np.argmax(fused_probs, axis=1)[0]
        confidence = np.max(fused_probs, axis=1)[0]
        
        return prediction, confidence, fused_probs[0], rf_prob[0], cnn_prob[0]
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None, None, None
def main():
    
    st.markdown('<div class="main-header">Threat Management in E-Commerce Platforms</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Integrating Explainable Artificial Intelligence in Anomaly Detection for Threat Management in E-Commerce Platforms</div>', unsafe_allow_html=True)
    
   
    with st.spinner("Loading AI models..."):
        scaler, pca, rf_model, cnn_lstm_model = load_models()
    
    if any(model is None for model in [scaler, pca, rf_model, cnn_lstm_model]):
        st.error(" Failed to load models. Please ensure all model files are available.")
        return
    
    st.success(" All models loaded successfully!")
    
    
    with st.sidebar:
        st.header(" Model Information")
        st.info("""
        **XAI Architecture:**
        - LIME(Random Forest Classifier)
        - GRAD-CAM(CNN-LSTM Neural Network)
        - XAI(Fusion)
        
        **Features:** 78 network flow characteristics
        **Classes:** Normal vs Malicious traffic
        """)
        
        st.header(" Quick Statistics")
        st.metric("Models Used", "3", "XAI")
        st.metric("Feature Dimensions", "78", "Original")
        st.metric("PCA Components", "Auto", "Optimized")
    
    
    tab1, tab2 = st.tabs([" Single Prediction", " Batch Analysis"])
    
    with tab1:
        st.header("Single Traffic Flow Analysis")
        
   
        features = create_feature_input_form()
        
     
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(" Analyze Traffic", use_container_width=True, type="primary"):
                with st.spinner(" Analyzing network traffic..."):
                    prediction, confidence, fused_prob, rf_prob, cnn_prob = predict_traffic(
                        features, scaler, pca, rf_model, cnn_lstm_model
                    )
                
                if prediction is not None:
                    
                    if prediction == 0:
                        st.markdown(f'''
                        <div class="prediction-normal">
                             NORMAL TRAFFIC DETECTED<br>
                            <small>Confidence: {confidence:.1%}</small>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="prediction-malicious">
                            MALICIOUS TRAFFIC DETECTED<br>
                            <small>Confidence: {confidence:.1%}</small>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                  
                    st.subheader(" Detailed Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "LIME",
                            f"{rf_prob[prediction]:.1%}",
                            f"{rf_prob[1] - rf_prob[0]:.1%}"
                        )
                    with col2:
                        st.metric(
                            "GRAD-CAM",
                            f"{fused_prob[prediction]:.1%}",
                            f"{fused_prob[1] - fused_prob[0]:.1%}"
                        )
                    with col3:
                        st.metric(
                            "XAI",
                            f"{confidence:.1%}",
                            f"{cnn_prob[1] - cnn_prob[0]:.1%}"
                        )
                    
                
                    fig_models = create_prediction_visualization(rf_prob, fused_prob, cnn_prob)
                    st.plotly_chart(fig_models, use_container_width=True)
                    
                    st.subheader(" Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Model': ['LIME', 'GRAD-CAM', 'XAI'],
                        'Normal': [rf_prob[0], fused_prob[0], cnn_prob[0]],
                        'Malicious': [rf_prob[1], fused_prob[1], cnn_prob[1]]
                    })
                    st.dataframe(prob_df, use_container_width=True)
    
        with tab2:
            st.header("Batch Analysis")
            st.info(" Upload a CSV file with network traffic features for batch processing")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(" Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    # Show dataset info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(df))
                    with col2:
                        st.metric("Features", len(df.columns))
                    with col3:
                        st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                    
                    if st.button(" Process Batch", type="primary"):
                        with st.spinner("Processing batch predictions..."):
                            
                            batch_results = process_batch_predictions(df, scaler, pca, rf_model, cnn_lstm_model)
                            
                            if batch_results is not None:
                                predictions, confidences, all_probs = batch_results
                                
                                
                                df_results = df.copy()
                                df_results['Prediction'] = ['Normal' if p == 0 else 'Malicious' for p in predictions]
                                df_results['Confidence'] = [f"{c:.1%}" for c in confidences]
                                df_results['Risk_Score'] = confidences
                                
                                st.success(f" Successfully processed {len(df)} traffic flows")
                                
                                
                                st.subheader(" Batch Analysis Summary")
                                
                                col1, col2, col3 = st.columns(3)
                                normal_count = sum(1 for p in predictions if p == 0)
                                malicious_count = len(predictions) - normal_count
                                avg_confidence = np.mean(confidences)
                                high_risk_count = sum(1 for c in confidences if c > 0.8)
                                
                                with col1:
                                    st.metric("Normal Traffic", normal_count, f"{normal_count/len(predictions):.1%}")
                                with col2:
                                    st.metric("Malicious Traffic", malicious_count, f"{malicious_count/len(predictions):.1%}")
                                with col3:
                                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                              
                                
                        
                                st.subheader(" Batch Results Visualization")
                                
                                
                                viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Distribution", "Risk Analysis", "Timeline", "Detailed Results"])
                                
                                with viz_tab1:
                                    
                                    fig_pie = px.pie(
                                        values=[normal_count, malicious_count],
                                        names=['Normal', 'Malicious'],
                                        title="Traffic Classification Distribution",
                                        color_discrete_map={'Normal': '#667eea', 'Malicious': '#f5576c'}
                                    )
                                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                    
                                   
                                    fig_hist = px.histogram(
                                        x=confidences,
                                        nbins=20,
                                        title="Confidence Score Distribution",
                                        labels={'x': 'Confidence Score', 'y': 'Count'},
                                        color_discrete_sequence=['#667eea']
                                    )
                                    st.plotly_chart(fig_hist, use_container_width=True)
                                
                                with viz_tab2:
                                    
                                    risk_data = pd.DataFrame({
                                        'Index': range(len(predictions)),
                                        'Risk_Score': confidences,
                                        'Prediction': ['Normal' if p == 0 else 'Malicious' for p in predictions]
                                    })
                                    
                                    fig_scatter = px.scatter(
                                        risk_data,
                                        x='Index',
                                        y='Risk_Score',
                                        color='Prediction',
                                        title="Risk Score Analysis Across Flows",
                                        color_discrete_map={'Normal': '#667eea', 'Malicious': '#f5576c'},
                                        hover_data={'Index': True, 'Risk_Score': ':.2%'}
                                    )
                                    fig_scatter.add_hline(y=0.8, line_dash="dash", line_color="red", 
                                                        annotation_text="High Risk Threshold")
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                                    
                                   
                                    
                                
                                with viz_tab3:
                                    
                                    if 'Flow Duration' in df.columns:
                                        timeline_data = pd.DataFrame({
                                            'Flow_Duration': df['Flow Duration'],
                                            'Prediction': ['Normal' if p == 0 else 'Malicious' for p in predictions],
                                            'Confidence': confidences
                                        })
                                        
                                        fig_timeline = px.scatter(
                                            timeline_data,
                                            x='Flow_Duration',
                                            y='Confidence',
                                            color='Prediction',
                                            title="Traffic Analysis by Flow Duration",
                                            color_discrete_map={'Normal': '#667eea', 'Malicious': '#f5576c'},
                                            labels={'Flow_Duration': 'Flow Duration (microseconds)', 'Confidence': 'Confidence Score'}
                                        )
                                        st.plotly_chart(fig_timeline, use_container_width=True)
                                    else:
                                        
                                        timeline_data = pd.DataFrame({
                                            'Flow_Index': range(len(predictions)),
                                            'Prediction': ['Normal' if p == 0 else 'Malicious' for p in predictions],
                                            'Confidence': confidences
                                        })
                                        
                                        fig_timeline = px.line(
                                            timeline_data,
                                            x='Flow_Index',
                                            y='Confidence',
                                            color='Prediction',
                                            title="Confidence Trends Across Flows",
                                            color_discrete_map={'Normal': '#667eea', 'Malicious': '#f5576c'}
                                        )
                                        st.plotly_chart(fig_timeline, use_container_width=True)
                                
                                with viz_tab4:
                                    
                                    st.subheader(" Detailed Results")
                                    
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        filter_prediction = st.selectbox("Filter by Prediction", ['All', 'Normal', 'Malicious'])
                                    with col2:
                                        min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.0, 0.1)
                                    with col3:
                                        show_features = st.checkbox("Show Original Features", False)
                                    
                                  
                                    filtered_df = df_results.copy()
                                    if filter_prediction != 'All':
                                        filtered_df = filtered_df[filtered_df['Prediction'] == filter_prediction]
                                    filtered_df = filtered_df[filtered_df['Risk_Score'] >= min_confidence]
                                    
                                    
                                    if show_features:
                                        st.dataframe(filtered_df, use_container_width=True)
                                    else:
                                        display_cols = ['Prediction', 'Confidence', 'Risk_Score']
                                        if len(filtered_df.columns) > 10:
                                            display_cols.extend(filtered_df.columns[:5].tolist())
                                        st.dataframe(filtered_df[display_cols], use_container_width=True)
                                    
                                    st.info(f"Showing {len(filtered_df)} out of {len(df_results)} flows")
                                
                              
                                st.subheader(" Download Results")
                                csv_download = df_results.to_csv(index=False)
                                st.download_button(
                                    label=" Download Results as CSV",
                                    data=csv_download,
                                    file_name=f"traffic_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    st.info("Please ensure your CSV file contains the required network traffic features.")
    
if __name__ == "__main__":
    main()