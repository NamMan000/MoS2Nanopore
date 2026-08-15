import streamlit as st
import hyperspy.api as hs
import matplotlib.pyplot as plt
import io
import tempfile
import os

from doping import extract_atoms_manual
from plotting_doping import plot_atoms


st.title("🔬 DM4 Image Processor")
st.write("Upload a .dm4 file to process and display the image.")


# --------------------------------------------------
# 1. File uploader
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a DM4 file",
    type=["dm4", "dm3", "tif", "tiff"]
)


# --------------------------------------------------
# 2. Load and display image
# --------------------------------------------------

if uploaded_file is not None:

    with st.spinner("Processing file..."):

        # Create temporary DM4 file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".dm4"
        ) as tmp_file:

            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Load DM4 with HyperSpy
            signal = hs.load(tmp_file_path)
            image_data = signal.data

        finally:
            # Remove temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)


    # --------------------------------------------------
    # 3. Display original image
    # --------------------------------------------------

    st.subheader("Original Image")

    fig_original, ax = plt.subplots()

    ax.imshow(image_data, cmap="gray")
    ax.axis("off")

    st.pyplot(fig_original)


    # --------------------------------------------------
    # 4. Download original image
    # --------------------------------------------------

    buf = io.BytesIO()

    fig_original.savefig(
        buf,
        format="png",
        bbox_inches="tight"
    )

    buf.seek(0)

    st.download_button(
        label="💾 Download Original Image",
        data=buf,
        file_name="processed_micrograph.png",
        mime="image/png"
    )


    # --------------------------------------------------
    # 5. Tungsten Doping button
    # --------------------------------------------------

    doping_but = st.button("Tungsten Doping")


    if doping_but:

        with st.spinner("Analyzing tungsten doping..."):

            # Run atom extraction
            moly_blobs, tungs_blobs = extract_atoms_manual(
                image_data,
                edge_padding=20
            )

            # Generate doping plot
            fig_doping = plot_atoms(
                moly_blobs,
                tungs_blobs,
                image_data
            )


        # --------------------------------------------------
        # 6. Display doping output
        # --------------------------------------------------

        st.subheader("Tungsten Doping Analysis")

        st.pyplot(fig_doping)