import requests
import streamlit as st


def show_results(tag_items):
    """
    Display the recommended tags in a "nice" layout.

    Parameters
    ----------
    tag_items : list[dict]
        The list of recommended tags with their scores. Each item is a dictionary
        containing the keys 'tag' and 'score'.

    Returns
    -------
    None
    """
    st.write("### Recommended Tags:")
    tag_cols = st.columns(8)  # Arrange the tags into columns
    for idx, tag_item in enumerate(tag_items):
        tag_col = tag_cols[idx % 8]

        tag_display = tag_item["tag"]
        if show_score:
            tag_display = f"{tag_display} (Score: {tag_item['score']:.3f})"

        tag_col.button(tag_display)


FASTAPI_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Tag Recommendation System", layout="wide")
st.title("Tag Recommendation System")

# Add image to sidebar from local file
st.sidebar.image("images/tumblr_icon.png")
st.sidebar.header("Settings")

# Number of recommended tags slider
num_tags = st.sidebar.slider(
    "Number of recommendations",
    min_value=1,
    max_value=30,
    value=10,
    help="Adjust the number of tags to recommend",
)

# Show score checkbox
show_score = st.sidebar.checkbox(
    "Show score", value=True, help="Toggle to show/hide recommendation scores"
)

# Text input for tags (overrides example if custom entered)
tags_input = st.text_input(
    "Enter tags (comma-separated)",
    value=None,
    help="Enter a list of tags separated by commas.",
)

if tags_input:
    # Prepare the request payload
    payload = {"tags": tags_input, "num_tags": num_tags}

    with st.spinner("Fetching recommendations..."):
        try:
            # Make the request to FastAPI
            response = requests.post(FASTAPI_URL, json=payload)
            response_data = response.json()

            if response.status_code == 200:
                st.success(response_data["status"]["message"])
                st.write(f"Input Tags: **{response_data['input_tags']}**")

                tags = response_data["tags"]
                if tags:
                    show_results(response_data["tags"])
                else:
                    st.write("No recommendations found.")
            else:
                st.error(f"Error: {response_data['status']['message']}")

        except Exception as e:
            st.error(f"Failed to connect to the recommendation service: {str(e)}")
else:
    st.error("Please enter some tags.")

st.sidebar.markdown(
    "Made with ❤️ by [George](https://www.linkedin.com/in/georgeperakis/)."
)
st.sidebar.markdown("[FastAPI Documentation](https://fastapi.tiangolo.com/)")
st.sidebar.markdown("[Streamlit Documentation](https://docs.streamlit.io/en/stable/)")
