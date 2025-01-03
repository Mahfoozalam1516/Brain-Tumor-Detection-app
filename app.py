import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pandas as pd
import datetime
import base64
from io import BytesIO
import cv2
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import sys

# Set page configuration
st.set_page_config(
    page_title="Advanced Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #4CAF50;
        color: white;
    }
    .upload-text {
        text-align: center;
        padding: 2rem;
        border: 2px dashed #ccc;
        border-radius: 10px;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Updated tab styling for better readability */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f8f9fa;
        border-radius: 4px;
        color: #2c3e50;
        font-weight: 500;
        border: 1px solid #e9ecef;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        border-color: #dee2e6;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3498db;
        color: white;
        border: none;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    /* Additional styles for better contrast */
    .stMarkdown {
        color: #f2f2f2;
    }
    h1, h2, h3 {
        color: #f2f2f2 !important;
    }
    .stAlert > div {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state with new variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'confidence_threshold': 0.7,
        'enable_image_enhancement': True,
        'save_results': True,
        'enable_notifications': False,
        'email_notifications': '',
        'dark_mode': False,
        'auto_enhance': True,
        'detection_sensitivity': 'medium'
    }
if 'comparison_images' not in st.session_state:
    st.session_state.comparison_images = []

# Load model function
def load_model():
    try:
        model = tf.keras.models.load_model('brain_tumor_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image function
def preprocess_image(image, enhance=True):
    try:
        if enhance:
            image = advanced_image_enhancement(image)
        image = image.resize((150, 150))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, image
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

# Predict tumor function
def predict_tumor(model, img_array):
    try:
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        confidence = np.max(predictions) * 100
        return class_names[predicted_class[0]], confidence, predictions
    except Exception as e:
        st.error(f"Error predicting tumor: {str(e)}")
        return None, None, None

# Enhanced image processing functions
def advanced_image_enhancement(image):
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_array, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply bilateral filter for edge preservation
    img_enhanced = cv2.bilateralFilter(img_enhanced, 9, 75, 75)
    
    # Sharpen image
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    img_enhanced = cv2.filter2D(img_enhanced, -1, kernel)
    
    return Image.fromarray(cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB))

# New function for image comparison
def compare_images(original, enhanced):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(enhanced)
    ax2.set_title('Enhanced Image')
    ax2.axis('off')
    return fig

# Enhanced PDF report generation
def create_advanced_report(result_data, image=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Enhanced title and header
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1a237e')
    )
    story.append(Paragraph("Advanced Brain Tumor Detection Report", title_style))
    story.append(Spacer(1, 12))
    
    # Add metadata
    metadata_style = ParagraphStyle(
        'Metadata',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey
    )
    story.append(Paragraph(f"Generated on: {result_data['date']}", metadata_style))
    story.append(Paragraph(f"Report ID: {datetime.datetime.now().strftime('%Y%m%d%H%M%S')}", metadata_style))
    story.append(Spacer(1, 20))
    
    # Add results with enhanced formatting
    result_style = ParagraphStyle(
        'Result',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor('#1b5e20') if result_data['prediction'] == 'No Tumor' else colors.HexColor('#b71c1c')
    )
    story.append(Paragraph(f"Diagnosis: {result_data['prediction']}", result_style))
    story.append(Paragraph(f"Confidence: {result_data['confidence']:.2f}%", result_style))
    
    # Add detailed analysis
    story.append(Spacer(1, 20))
    story.append(Paragraph("Detailed Analysis", styles['Heading2']))
    
    # Create a table for probability distribution
    data = [['Class', 'Probability']]
    for idx, prob in enumerate(['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']):
        data.append([prob, f"{result_data['raw_predictions'][0][idx] * 100:.2f}%"])
    
    table = Table(data, colWidths=[200, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BOX', (0, 0), (-1, -1), 2, colors.black),
        ('LINEBELOW', (0, 0), (-1, 0), 2, colors.black),
    ]))
    story.append(table)
    
    # Add recommendations
    story.append(Spacer(1, 20))
    story.append(Paragraph("Recommendations:", styles['Heading2']))
    if result_data['prediction'] != 'No Tumor':
        recommendations = [
            "1. Consult with a neurologist immediately",
            "2. Schedule additional diagnostic tests (MRI with contrast, CT scan)",
            "3. Maintain a copy of this report for medical records",
            "4. Consider seeking a second opinion",
            "5. Schedule regular follow-up appointments"
        ]
    else:
        recommendations = [
            "1. Continue regular health check-ups",
            "2. Maintain a healthy lifestyle",
            "3. Report any new symptoms to your healthcare provider",
            "4. Schedule next screening as recommended by your doctor"
        ]
    
    for rec in recommendations:
        story.append(Paragraph(rec, styles['Normal']))
    
    # Add disclaimer
    story.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey
    )
    story.append(Paragraph("DISCLAIMER: This report is generated by an AI-based system and should not be used as the sole basis for medical decisions. Always consult with qualified healthcare professionals for proper diagnosis and treatment.", disclaimer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# New function for email notifications
def send_email_notification(recipient, result_data, pdf_report):
    try:
        # Email configuration
        sender_email = "your-email@example.com"  # Configure your email
        password = "your-password"  # Configure your password
        
        msg = MIMEMultipart()
        msg['Subject'] = f"Brain Tumor Detection Results - {result_data['date']}"
        msg['From'] = sender_email
        msg['To'] = recipient
        
        # Email body
        body = f"""
        Brain Tumor Detection Results
        
        Date: {result_data['date']}
        Prediction: {result_data['prediction']}
        Confidence: {result_data['confidence']:.2f}%
        
        Please find the detailed report attached.
        
        Note: This is an automated notification. Please do not reply to this email.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF report
        pdf_attachment = MIMEApplication(pdf_report.read(), _subtype='pdf')
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename='brain_tumor_report.pdf')
        msg.attach(pdf_attachment)
        
        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
        
        return True
    except Exception as e:
        st.error(f"Failed to send email notification: {str(e)}")
        return False

# Enhanced settings UI
def show_advanced_settings():
    st.sidebar.title("Advanced Settings")
    
    # Analysis Settings
    st.sidebar.subheader("Analysis Settings")
    st.session_state.settings['confidence_threshold'] = st.sidebar.slider(
        "Confidence Threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=70
    )
    st.session_state.settings['detection_sensitivity'] = st.sidebar.select_slider(
        "Detection Sensitivity",
        options=['low', 'medium', 'high'],
        value='medium'
    )
    
    # Image Enhancement Settings
    st.sidebar.subheader("Image Enhancement")
    st.session_state.settings['enable_image_enhancement'] = st.sidebar.checkbox(
        "Enable Image Enhancement",
        value=True
    )
    st.session_state.settings['auto_enhance'] = st.sidebar.checkbox(
        "Auto-apply Enhancement",
        value=True
    )
    
    # Notification Settings
    st.sidebar.subheader("Notifications")
    st.session_state.settings['enable_notifications'] = st.sidebar.checkbox(
        "Enable Email Notifications",
        value=False
    )
    if st.session_state.settings['enable_notifications']:
        st.session_state.settings['email_notifications'] = st.sidebar.text_input(
            "Notification Email",
            value=st.session_state.settings.get('email_notifications', '')
        )
    
    # Display Settings
    st.sidebar.subheader("Display Settings")
    st.session_state.settings['dark_mode'] = st.sidebar.checkbox(
        "Dark Mode",
        value=False
    )
    
    # Data Management
    st.sidebar.subheader("Data Management")
    st.session_state.settings['save_results'] = st.sidebar.checkbox(
        "Save Results Locally",
        value=True
    )
    
    if st.sidebar.button("Reset All Settings"):
        st.session_state.settings = {
            'confidence_threshold': 0.7,
            'enable_image_enhancement': True,
            'save_results': True,
            'enable_notifications': False,
            'email_notifications': '',
            'dark_mode': False,
            'auto_enhance': True,
            'detection_sensitivity': 'medium'
        }
        st.success("Settings reset to default values!")

# Enhanced main function
def main():
    # Show settings in sidebar
    show_advanced_settings()
    
    # Create tabs with new features
    tabs = st.tabs([
        "Single Image Analysis", 
        "Batch Processing", 
        "History & Analytics",
        "Image Comparison",
        "Settings & Help"
    ])
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check if 'brain_tumor_model.h5' exists in the correct location.")
        return
    
    # Tab 1: Enhanced Single Image Analysis
    with tabs[0]:
        st.title("🧠 Advanced Brain Tumor Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Upload MRI Scan")
            uploaded_file = st.file_uploader(
                "Choose an MRI image...", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear MRI scan image for analysis"
            )
            
            if uploaded_file is not None:
                try:
                    # Display original image
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Original MRI Scan', use_column_width=True)
                    
                    # Show enhanced image if enabled
                    if st.session_state.settings['enable_image_enhancement']:
                        enhanced_image = advanced_image_enhancement(image)
                        st.image(enhanced_image, caption='Enhanced MRI Scan', use_column_width=True)
                        # Add image comparison
                        if st.button('Compare Original vs Enhanced'):
                            comparison_fig = compare_images(image, enhanced_image)
                            st.pyplot(comparison_fig)
                    
                    if st.button('Analyze Image', key='single_analysis'):
                        with st.spinner('Performing detailed analysis...'):
                            # Preprocess image and make prediction
                            processed_image, _ = preprocess_image(
                                image, 
                                enhance=st.session_state.settings['enable_image_enhancement']
                            )
                            predicted_class, confidence, raw_predictions = predict_tumor(model, processed_image)
                            
                            # Prepare result data
                            result_data = {
                                'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'prediction': predicted_class,
                                'confidence': confidence,
                                'raw_predictions': raw_predictions.tolist(),
                                'settings_used': st.session_state.settings.copy()
                            }
                            
                            # Store result in history
                            st.session_state.history.append(result_data)
                            
                            # Save result if enabled
                            if st.session_state.settings['save_results']:
                                save_result(result_data)
                            
                            # Display results in second column
                            with col2:
                                st.markdown("### Detailed Analysis Results")
                                
                                # Enhanced result display
                                result_color = "#28a745" if predicted_class == "No Tumor" else "#dc3545"
                                st.markdown(f"""
                                <div style='background-color: {"#f8fff8" if predicted_class == "No Tumor" else "#fff8f8"}; 
                                            padding: 20px; 
                                            border-radius: 10px; 
                                            border-left: 5px solid {result_color};
                                            margin-bottom: 20px;'>
                                    <h2 style='color: {result_color}; margin-top: 0;'>
                                        {predicted_class}
                                    </h2>
                                    <p style='font-size: 24px; margin-bottom: 10px;'>
                                        Confidence: {confidence:.2f}%
                                    </p>
                                    <p style='color: #666; margin-bottom: 0;'>
                                        Analysis completed at {result_data['date']}
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Generate enhanced PDF report
                                report_buffer = create_advanced_report(result_data, image)
                                
                                # Offer report download
                                col2_1, col2_2 = st.columns(2)
                                with col2_1:
                                    st.download_button(
                                        label="📄 Download Detailed Report",
                                        data=report_buffer,
                                        file_name=f"brain_tumor_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf"
                                    )
                                
                                # Email notification if enabled
                                with col2_2:
                                    if st.session_state.settings['enable_notifications']:
                                        if st.button('📧 Send Report via Email'):
                                            if send_email_notification(
                                                st.session_state.settings['email_notifications'],
                                                result_data,
                                                report_buffer
                                            ):
                                                st.success("Report sent successfully!")
                                
                                # Display enhanced visualization
                                st.markdown("### Detailed Probability Analysis")
                                
                                # Create tabs for different visualizations
                                viz_tabs = st.tabs(["Bar Chart", "Radar Chart", "Confidence Gauge"])
                                
                                with viz_tabs[0]:
                                    # Enhanced bar chart
                                    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                                    probabilities = raw_predictions[0] * 100
                                    
                                    fig = go.Figure(data=[
                                        go.Bar(
                                            x=class_names,
                                            y=probabilities,
                                            marker_color=['#007bff' if name != predicted_class else '#28a745' 
                                                        for name in class_names],
                                            text=[f'{prob:.1f}%' for prob in probabilities],
                                            textposition='auto',
                                        )
                                    ])
                                    fig.update_layout(
                                        title="Prediction Probabilities",
                                        yaxis_title="Probability (%)",
                                        xaxis_title="Class",
                                        showlegend=False,
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with viz_tabs[1]:
                                    # Radar chart
                                    fig = go.Figure(data=go.Scatterpolar(
                                        r=probabilities,
                                        theta=class_names,
                                        fill='toself',
                                        marker_color='rgb(40, 167, 69)'
                                    ))
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 100]
                                            )
                                        ),
                                        showlegend=False,
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with viz_tabs[2]:
                                    # Confidence gauge
                                    fig = go.Figure(go.Indicator(
                                        mode="gauge+number",
                                        value=confidence,
                                        domain={'x': [0, 1], 'y': [0, 1]},
                                        gauge={
                                            'axis': {'range': [0, 100]},
                                            'bar': {'color': result_color},
                                            'steps': [
                                                {'range': [0, 50], 'color': '#ff4444'},
                                                {'range': [50, 75], 'color': '#ffbb33'},
                                                {'range': [75, 100], 'color': '#00C851'}
                                            ],
                                            'threshold': {
                                                'line': {'color': "red", 'width': 4},
                                                'thickness': 0.75,
                                                'value': st.session_state.settings['confidence_threshold']
                                            }
                                        },
                                        title={'text': "Confidence Level"}
                                    ))
                                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Please try uploading a different image or contact support if the issue persists.")
    
    # Tab 2: Enhanced Batch Processing
    with tabs[1]:
        st.title("Advanced Batch Processing")
        
        # Batch upload options
        upload_option = st.radio(
            "Choose upload method:",
            ["Individual Files", "ZIP Archive"],
            horizontal=True
        )
        
        if upload_option == "Individual Files":
            uploaded_files = st.file_uploader(
                "Upload multiple MRI images...",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True
            )
        else:
            zip_file = st.file_uploader("Upload ZIP file containing MRI images...", type=['zip'])
            if zip_file:
                with zipfile.ZipFile(zip_file) as z:
                    uploaded_files = [
                        BytesIO(z.read(fname)) 
                        for fname in z.namelist() 
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
            else:
                uploaded_files = []
        
        if uploaded_files:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"Total files: {len(uploaded_files)}")
            with col2:
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=len(uploaded_files),
                    value=min(10, len(uploaded_files))
                )
            with col3:
                process_button = st.button('Start Batch Processing')
            
            if process_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                batch_results = []
                
                # Process images in batches
                for i in range(0, len(uploaded_files), batch_size):
                    batch = uploaded_files[i:i + batch_size]
                    status_text.text(f"Processing batch {i//batch_size + 1}/{(len(uploaded_files)-1)//batch_size + 1}...")
                    
                    for file in batch:
                        try:
                            image = Image.open(file)
                            processed_image, _ = preprocess_image(
                                image,
                                enhance=st.session_state.settings['enable_image_enhancement']
                            )
                            predicted_class, confidence, raw_predictions = predict_tumor(model, processed_image)
                            
                            batch_results.append({
                                'filename': getattr(file, 'name', 'unknown'),
                                'prediction': predicted_class,
                                'confidence': confidence,
                                'raw_predictions': raw_predictions.tolist()
                            })
                            
                        except Exception as e:
                            st.error(f"Error processing {getattr(file, 'name', 'unknown')}: {str(e)}")
                        
                        progress_bar.progress((len(batch_results)) / len(uploaded_files))
                
                if batch_results:
                    # Create detailed results DataFrame
                    df = pd.DataFrame(batch_results)
                    
                    # Display summary statistics
                    st.subheader("Batch Analysis Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Total Processed",
                            len(df),
                            f"+{len(df)} files"
                        )
                    
                    with col2:
                        tumor_count = len(df[df['prediction'] != 'No Tumor'])
                        st.metric(
                            "Tumor Detected",
                            tumor_count,
                            f"{(tumor_count/len(df)*100):.1f}%"
                        )
                    
                    with col3:
                        avg_conf = df['confidence'].mean()
                        st.metric(
                            "Average Confidence",
                            f"{avg_conf:.1f}%",
                            f"{df['confidence'].std():.1f}% σ"
                        )
                    
                    # Display detailed results
                    st.subheader("Detailed Results")
                    st.dataframe(df)
                    
                    # Create visualization of batch results
                    st.subheader("Batch Analysis Visualization")
                    viz_tabs = st.tabs(["Distribution", "Confidence Analysis"])
                    
                    with viz_tabs[0]:
                        fig = px.pie(
                            df,
                            names='prediction',
                            title='Distribution of Predictions',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig)
                    
                    with viz_tabs[1]:
                        fig = px.box(
                            df,
                            x='prediction',
                            y='confidence',
                            title='Confidence Distribution by Prediction',
                            color='prediction'
                        )
                        st.plotly_chart(fig)
                    
                    # Export options
                    st.subheader("Export Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export as CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results (CSV)",
                            data=csv,
                            file_name=f"batch_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Export as Excel
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer) as writer:
                            df.to_excel(writer, index=False)
                        buffer.seek(0)
                        st.download_button(
                            label="Download Results (Excel)",
                            data=buffer,
                            file_name=f"batch_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
    
    # Tab 3: Enhanced History & Analytics
    with tabs[2]:
        st.title("Advanced Analysis History & Analytics")
        
        if st.session_state.history:
            # Convert history to DataFrame
            history_df = pd.DataFrame(st.session_state.history)
            
            # Time series analysis
            st.subheader("Analysis Timeline")
            timeline_fig = px.line(
                history_df,
                x='date',
                y='confidence',
                color='prediction',
                title='Confidence Trends Over Time'
            )
            st.plotly_chart(timeline_fig)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Analyses",
                    len(history_df),
                    f"+{len(history_df)} cases"
                )
            
            with col2:
                tumor_rate = len(history_df[history_df['prediction'] != 'No Tumor']) / len(history_df) * 100
                st.metric(
                    "Tumor Detection Rate",
                    f"{tumor_rate:.1f}%",
                    f"{tumor_rate - 50:.1f}% from baseline"
                )
            
            with col3:
                avg_confidence = history_df['confidence'].mean()
                st.metric(
                    "Average Confidence",
                    f"{avg_confidence:.1f}%",
                    f"±{history_df['confidence'].std():.1f}%"
                )
            
            with col4:
                recent_avg = history_df.tail(10)['confidence'].mean()
                st.metric(
                    "Recent Avg (Last 10)",
                    f"{recent_avg:.1f}%",
                    f"{recent_avg - avg_confidence:.1f}%"
                )
            
            # Advanced visualizations
            viz_tabs = st.tabs(["Distribution Analysis", "Confidence Analysis", "Time Analysis"])
            
            with viz_tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        history_df,
                        names='prediction',
                        title='Overall Distribution of Predictions'
                    )
                    st.plotly_chart(fig)
                
                with col2:
                    # Create confusion matrix-style visualization
                    prediction_counts = history_df['prediction'].value_counts()
                    fig = px.bar(
                        prediction_counts,
                        title='Prediction Distribution Analysis'
                    )
                    st.plotly_chart(fig)
            
            with viz_tabs[1]:
                # Confidence distribution analysis
                fig = px.histogram(
                    history_df,
                    x='confidence',
                    color='prediction',
                    marginal='box',
                    title='Confidence Distribution Analysis'
                )
                st.plotly_chart(fig)
            
            with viz_tabs[2]:
                # Time-based analysis
                daily_counts = history_df.copy()
                daily_counts['date'] = pd.to_datetime(daily_counts['date'])
                daily_counts = daily_counts.resample('D', on='date').size()
                
                fig = px.line(
                    daily_counts,
                    title='Daily Analysis Volume',
                    labels={'value': 'Number of Analyses', 'date': 'Date'}
                )
                st.plotly_chart(fig)
            
            # Export options
            st.subheader("Export History")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download History (CSV)",
                    data=csv,
                    file_name=f"analysis_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                if st.button("Clear History"):
                    st.session_state.history = []
                    st.experimental_rerun()
        
        else:
            st.info("No analysis history available yet. Start by analyzing some images!")
    
    # Tab 4: Image Comparison Tool
    with tabs[3]:
        st.title("Image Comparison Tool")
        
        st.markdown("""
        This tool allows you to compare different MRI scans and their analysis results.
        Upload multiple images to perform comparative analysis.
        """)
        
        comparison_files = st.file_uploader(
            "Upload images for comparison",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key='comparison_upload'
        )
        
        if comparison_files:
            if len(comparison_files) < 2:
                st.warning("Please upload at least 2 images for comparison.")
            else:
                st.subheader("Comparative Analysis")
                
                # Process all images
                comparison_results = []
                for file in comparison_files:
                    image = Image.open(file)
                    processed_image, _ = preprocess_image(
                        image,
                        enhance=st.session_state.settings['enable_image_enhancement']
                    )
                    predicted_class, confidence, raw_predictions = predict_tumor(model, processed_image)
                    
                    comparison_results.append({
                        'filename': file.name,
                        'image': image,
                        'prediction': predicted_class,
                        'confidence': confidence,
                        'raw_predictions': raw_predictions
                    })
                
                # Display comparison grid
                cols = st.columns(len(comparison_results))
                for idx, (col, result) in enumerate(zip(cols, comparison_results)):
                    with col:
                        st.image(result['image'], caption=f"Image {idx + 1}")
                        st.markdown(f"""
                        **Prediction:** {result['prediction']}  
                        **Confidence:** {result['confidence']:.1f}%
                        """)
                
                # Comparative visualization
                st.subheader("Comparative Analysis")
                
                # Create comparative bar chart
                comparison_data = pd.DataFrame([
                    {
                        'filename': r['filename'],
                        'prediction': r['prediction'],
                        'confidence': r['confidence']
                    } for r in comparison_results
                ])
                
                fig = px.bar(
                    comparison_data,
                    x='filename',
                    y='confidence',
                    color='prediction',
                    title='Confidence Comparison'
                )
                st.plotly_chart(fig)
                
                # Export comparison results
                if st.button("Export Comparison Report"):
                    # Generate comparative report
                    report_buffer = create_advanced_report(comparison_results[0])  # Example for first result
                    st.download_button(
                        label="Download Comparison Report",
                        data=report_buffer,
                        file_name=f"comparison_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
    
    # Tab 5: Settings & Help
    with tabs[4]:
        st.title("Settings & Help")
        
        # Settings management
        st.subheader("Settings Management")
        
        # Display current settings
        st.json(st.session_state.settings)
        
        # Export/Import settings
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Settings"):
                settings_json = json.dumps(st.session_state.settings, indent=2)
                st.download_button(
                    label="Download Settings",
                    data=settings_json,
                    file_name=f"settings_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_settings = st.file_uploader("Import Settings", type=['json'])
            if uploaded_settings:
                try:
                    new_settings = json.load(uploaded_settings)
                    st.session_state.settings.update(new_settings)
                    st.success("Settings imported successfully!")
                except Exception as e:
                    st.error(f"Error importing settings: {str(e)}")
        
        # Help section
        st.subheader("Help & Documentation")
        
        with st.expander("Quick Start Guide"):
            st.markdown("""
            1. Upload an MRI scan image using the upload button
            2. Choose whether to enhance the image
            3. Click 'Analyze Image' to get results
            4. View detailed analysis and download reports
            """)
        
        with st.expander("Image Requirements"):
            st.markdown("""
            - Supported formats: JPG, JPEG, PNG
            - Recommended resolution: 150x150 pixels or higher
            - Clear, focused MRI scans
            - Proper orientation (axial view preferred)
            """)
        
        with st.expander("Understanding Results"):
            st.markdown("""
            The system can detect four categories:
            - Glioma
            - Meningioma
            - No Tumor
            - Pituitary
            
            Confidence scores indicate the model's certainty in its prediction.
            """)
        
        with st.expander("Troubleshooting"):
            st.markdown("""
            Common issues and solutions:
            1. Image upload fails
               - Check file format and size
               - Try a different browser
            2. Low confidence scores
               - Ensure image quality
               - Try image enhancement
            3. Processing errors
               - Refresh the page
               - Clear browser cache
            """)
        
        # System information
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            - Model Version: 1.0.0
            - Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d')}
            - Python Version: {sys.version.split()[0]}
            """)
        
        with col2:
            st.markdown(f"""
            - TensorFlow Version: {tf.__version__}
            - OpenCV Version: {cv2.__version__}
            - Streamlit Version: {st.__version__}
            """)

if __name__ == "__main__":
    main()