import streamlit as 
import hyperspy.api as hs
import matplotlib.pyplot as plt
import io


st.title("🔬 DM4 Image Processor")
st.write("Upload a .dm4 file to process and display the image.")
uploaded_file = st.file_uploader("Choose a DM4 file", type=["dm4"])


if uploaded_file is not None:
    with st.spinner("Processing file..."):
           # 2. Process the DM4 file
        signal = hs.load(uploaded_file)
        image_data = signal.data
        
        # --- DO YOUR PROCESSING HERE ---
        # e.g., processed_data = image_data * 2 
        
        # 3. Create the output plot/image
        fig, ax = plt.subplots()
        ax.imshow(image_data, cmap='gray')
        ax.axis('off')

        st.pyplot(fig)
        
        # 4. Make the image downloadable
        # Save the plot to a memory buffer so the user can download it
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="💾 Download Processed Image",
            data=buf,
            file_name="processed_micrograph.png",
            mime="image/png"
        )