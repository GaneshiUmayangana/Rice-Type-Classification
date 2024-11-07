import streamlit as st

# URL for the background image
background_image_url = "https://www.suncakemom.com/wp-content/uploads/2024/02/All-about-rice-types-5-650x434.webp"

# Define the chatbot responses
responses = {
    "jasmine rice": "Jasmine Rice is a fragrant, long-grain rice commonly used in Southeast Asian dishes.",
    "basmati rice": "Basmati Rice is known for its long, fluffy grains and is used in Indian and Middle Eastern cuisines.",
    "arborio rice": "Arborio Rice is a short-grain rice used to make creamy risotto.",
    "sushi rice": "Sushi Rice is seasoned rice that becomes sticky and is used in sushi dishes.",
    "what is jasmine rice": "Jasmine Rice is a fragrant, long-grain rice commonly used in Southeast Asian dishes.",
    "how to cook basmati rice": "To cook Basmati Rice, rinse it first to remove excess starch, then cook it in water (1:1.5 ratio) and simmer for 10-15 minutes.",
    "is arborio rice good for risotto": "Yes, Arborio Rice is perfect for risotto due to its high starch content that creates a creamy texture.",
    "what is sushi rice made of": "Sushi Rice is made of short-grain rice seasoned with vinegar, sugar, and salt, giving it a slightly sweet and tangy flavor.",
    "what is the difference between basmati and jasmine rice": "Basmati Rice is nutty and long-grain, while Jasmine Rice is fragrant and sticky, often used in Thai and Southeast Asian dishes.",
    "how do i store sushi rice": "Sushi Rice should be stored at room temperature in an airtight container to maintain its texture and flavor.",
    "can i use basmati rice for sushi": "While it's not ideal, you can use Basmati Rice for sushi in a pinch, but it will not have the same sticky texture as Sushi Rice."
}

def get_chatbot_response(user_input):
    """
    Function to get a response based on user input.
    It checks for keywords in the input and provides a matching response.
    """
    user_input = user_input.lower()
    
    # Check for direct keyword matches in responses
    for keyword in responses:
        if keyword in user_input:
            return responses[keyword]
    
    # Default response if no keyword matches
    return "Sorry, I don't have an answer for that. Please ask another question related to rice."

def show_explore_page():
    # Custom CSS for background image and label styles with increased font sizes
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('{background_image_url}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        label {{
            font-size: 24px !important;
            font-weight: bold;
            color: black !important;
            display: block;
            padding: 15px;
            border-radius: 10px;
            background-color: #F5F5DC;
            text-align: center;
        }}
        h1 {{
            font-size: 36px !important;
            color: #333 !important;
            text-align: center;
        }}
        h2 {{
            font-size: 30px !important;
            color: #333 !important;
        }}
        h3 {{
            font-size: 28px !important;
            color: #333 !important;
        }}
        p {{
            font-size: 20px !important;
            line-height: 1.6;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title of the page
    st.title("Explore Rice Varieties")

    # Introduction
    st.write("Welcome to the Rice Varieties Chatbot! Explore different rice varieties and ask any questions you may have.")
    
    # Displaying rice varieties with descriptions and images
    st.header("Popular Rice Varieties")

    # Jasmine Rice
    st.subheader("Jasmine Rice")
    st.image("jasmine.jpg", caption="Jasmine Rice")
    st.write("""
    **Jasmine Rice** is a fragrant, long-grain rice often used in Thai and Southeast Asian dishes. 
    It has a soft, sticky texture when cooked and a floral aroma, pairing well with curries and stir-fries.
    """)

    # Basmati Rice
    st.subheader("Basmati Rice")
    st.image("basmati.jpg", caption="Basmati Rice")
    st.write("""
    **Basmati Rice** is known for its long, fluffy grains and nutty flavor. Commonly used in Indian and Middle Eastern cuisines, 
    it is perfect for dishes like biryanis and pilafs and pairs well with rich curries.
    """)

    # Arborio Rice
    st.subheader("Arborio Rice")
    st.image("arborio.jpg", caption="Arborio Rice")
    st.write("""
    **Arborio Rice** is a short-grain rice primarily used for making risotto. It absorbs liquids well and releases starch, 
    creating a creamy texture that is ideal for risotto and other Italian dishes.
    """)

    # Sushi Rice
    st.subheader("Sushi Rice")
    st.image("sushi.jpg", caption="Sushi Rice")
    st.write("""
    **Sushi Rice** is a Japanese short-grain rice that becomes sticky when cooked. It is seasoned with vinegar, 
    sugar, and salt, making it ideal for sushi and other Japanese dishes.
    """)

    # Chatbot interaction
    st.header("Rice Chatbot")
    st.write("Have questions about rice varieties? Ask below!")

    # Input from user
    user_input = st.text_input("Type your question about rice varieties:")
    if user_input:
        with st.spinner("Thinking..."):
            response = get_chatbot_response(user_input)
        st.write("Chatbot Response:")
        st.write(response)

# Call the function to run the explore page
if __name__ == "__main__":
    show_explore_page()
