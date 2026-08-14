import streamlit as st
import hyperspy.api as hs
import matplotlib.pyplot as plt
import io
import tempfile
import os

from processor import analyze_dm4_image

st.title("🔬 DM4 Image Processor")
st.write("Upload a .dm4 file to process and display the image.")

# 1. File Uploader Widget
uploaded_file = st.file_uploader("Choose a DM4 file", type=["dm4"])

if uploaded_file is not None:
    with st.spinner("Processing file..."):
        
        # --- THE FIX: Create a temporary file on disk ---
        # "delete=False" keeps the file open so HyperSpy can access it by its path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dm4") as tmp_file:
            tmp_file.write(uploaded_file.getvalue()) # Write the uploaded memory bytes to disk
            tmp_file_path = tmp_file.name # Get the literal file path (e.g., /tmp/xyz.dm4)

        try:
            # 2. Pass the temporary path string to HyperSpy
            signal = hs.load(tmp_file_path)
            image_data = signal.data
            
            # 3. Create the output plot/image
            fig, ax = plt.subplots()
            ax.imshow(image_data, cmap='gray')
            ax.axis('off')
            
            # Display the image on the web app page
            st.pyplot(fig)
            
            # 4. Make the image downloadable
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="💾 Download Processed Image",
                data=buf,
                file_name="processed_micrograph.png",
                mime="image/png"
            )
            
        finally:
            # Clean up the temporary file from the server disk when finished
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
