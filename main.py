import streamlit as st

# Define function for each page
def home():
    st.title("Home Page")
    st.write("Welcome to the home page!")

def about():
    st.title("About Page")
    st.write("This is the about page.")

def contact():
    st.title("Contact Page")
    st.write("You can contact us here.")

# Define main function to manage navigation
def main():
    pages = {
        "Home": home,
        "About": about,
        
    }

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()

if __name__ == "__main__":
    main()
