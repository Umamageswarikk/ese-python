
import streamlit as st
import pandas as pd
from PIL import Image
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from PIL import Image
import numpy as np


# Load data
data = pd.read_csv(r"C:\MCA\3rd trimester\advanced python\ese\WomensClothingE-CommerceReviews.csv")

data=data.dropna()

general=(data['Division Name']=="General")
dg=data[general]
dg['Review Text']

general_petite=(data['Division Name']=="General Petite")
dp=data[general_petite]
dp['Review Text']

Initmates=(data['Division Name']=="Initmates")
Init=data[Initmates]
Init['Review Text']


# Sidebar selection
st.sidebar.title('ESE STREAMLIT APPLICATION')
selected_option = st.sidebar.radio("Select Visualization", ("3D PLOT VISUALIZATION", "IMAGE PROCESSING", "TEXT SIMILARITY ANALYSIS"))


st.title("ESE STREAMLIT APPLICATION")

if selected_option == "3D PLOT VISUALIZATION":

    st.subheader("3D PLOT VISUALIZATION")

    st.image(Image.open(r"C:\MCA\3rd trimester\advanced python\streamlit\shopping-sale.gif"),width=230) 

 

    st.subheader("Data Frame")
    st.write(data)


    st.subheader("3D Plot")

         
    x=st.selectbox("select a feature for x axis",data.columns)
    y=st.selectbox("select a feature for y axis",data.columns)
    z=st.selectbox("select a feature for z axis",data.columns)

    x_data=data[x]
    y_data=data[y]
    z_data=data[z]

   

    # Create a 3D plot
    
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(111,projection='3d')

    fig = plt.figure()
    ax = plt.axes(projection="3d")


    ax.scatter3D(x_data,y_data,z_data,cmap='cividis',c=y_data) 
    ax.set_xlabel(f'{x}')
    ax.set_ylabel(f'{y}')
    ax.set_zlabel(f'{z}')
    # plt.show()
    st.pyplot(fig)




elif selected_option == "TEXT ANALYSIS":

    
    st.subheader("Text ANALYSIS")
    data['Review Text']=data['Review Text'].str.lower()

    # Tokenize the text column
    data['tokens'] = data['Review Text'].apply(word_tokenize)

    # Remove stop words
    stop_words=set(stopwords.words("english"))



    # Filter stopwords from each tokenized text 

    filtered_texts = []
    for token_list in data['tokens']:
        filtered_word = []
        for word in token_list:
            if word.casefold() not in stop_words:
                filtered_word.append(word)
        filtered_texts.append(filtered_word)

    data['filtered_list'] = filtered_texts

    # the tokens are in list format so we use two lists

    print("Before filtering:", data['tokens'].apply(len).sum())
    print("After filtering:", data['filtered_list'].apply(len).sum())
    #print("After stop word removal : \n",data.tail(5))

    # lemmatisation
    lemmatizer=WordNetLemmatizer()
    lemmatized_Words=[] # for all rows
    for filtered_word in data['filtered_list']:
        lemmatized_text=[] # for each row
        for word in filtered_word:
            # Remove punctuation or special characters from lemmatized data
            word = ''.join(char for char in word if char.isalpha())
            lemmatized_text.append(lemmatizer.lemmatize(word))
        lemmatized_Words.append(lemmatized_text)

    data['lemmatized_words']=lemmatized_Words    
    data['lemmatized_words']    
    print("After Lemmatization : \n ")
    st.write(Review_Text=data.tail(5))




    # general
    text_general = dg['Review Text']
    # Write the general reviews to Text_general file
    with open('General.txt', 'w', encoding='utf-8') as file:
        for review in text_general:
            file.write(review + '\n')

    # writing petite reviews
    text_petite = dp['Review Text']
    with open('petite.txt', 'w', encoding='utf-8') as file:
        for review in text_petite:
            file.write(review + '\n')

    # Initmates
    text_initmates = Init['Review Text']
    with open('Initmates.txt', 'w', encoding='utf-8') as file:
        for review in text_initmates:
            file.write(review + '\n')

    f=open("General.txt")
    data1=f.read()

    f=open("petite.txt")
    data2=f.read()

    f=open("Initmates.txt")
    data3=f.read()

   

    selected_text = st.selectbox("Select a division", ("General", "General Petite", "Initmates"))

    if selected_text == "General":
        st.subheader("General Data")
        st.write(data1)
        selected_data = data[general]  # Filter data for General division
    elif selected_text == "General Petite":
        st.subheader("General Petite Data")
        st.write(data2)
        selected_data = data[general_petite]  # Filter data for General Petite division
    elif selected_text == "Initmates":
        st.subheader("Initmates Data")
        st.write(data3)
        selected_data = data[Initmates]  # Filter data for Initmates division

    st.write(selected_data)


    # Tokenize and remove stopwords
    stop_words = set(stopwords.words("english"))

    def preprocess_text(data):
        given_text=word_tokenize(data1)
        stop_words=set(stopwords.words("english"))

        filtered_list=[]
        for word in given_text:
            if word.casefold() not in stop_words:
                filtered_list.append(word)
        filter=' '.join(map(str,filtered_list))        

        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        lemmatized_text = [lemmatizer.lemmatize(word) for word in filter]
        lem=' '.join(map(str,filter))  
        
        return {
            'word_tokenizer': word,
            'filtered_text': filter,
            'lemmatized_text': lem
        }

    # Preprocess sentences
    tokens1 = set(preprocess_text(data1))
    tokens2 = set(preprocess_text(data2))
    tokens3=set(preprocess_text(data3))


    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Create TF-IDF vectors for data1
    vector1 = vectorizer.fit_transform([data1])

    # Create TF-IDF vectors for data2 using the same vectorizer instance
    vector2 = vectorizer.fit_transform([data2])

    vector3=vectorizer.fit_transform([data3])


    def jaccard_similarity(set1,set2):
        intersection=len(set1.intersection(set2))
        union=len(set1.union(set2))
        return intersection/union

    # cal jaccard similarity
    jaccard_similariy_score1=jaccard_similarity(tokens1,tokens2)
    st.write(f"jaccard similarity between general and petite : {jaccard_similariy_score1}") 

    # cal jaccard similarity
    jaccard_similariy_score2=jaccard_similarity(tokens3,tokens2)
    st.write(f"jaccard similarity between initmates and petite : {jaccard_similariy_score2}") 

    


elif selected_option == "IMAGE PROCESSING":
    

    # Function to resize image
    def resize_image(image, width, height):
        resized_image = image.resize((width, height))
        return resized_image

    # Function to convert image to grayscale
    def grayscale_image(image):
        grayscale_image = image.convert('L')
        return grayscale_image

    # Function to crop image
    def crop_image(image, left, top, right, bottom):
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

    # Streamlit app
    def main():
        st.title("Image Processing Techniques")

        # Upload image
        uploaded_image = st.file_uploader(r"C:\MCA\3rd trimester\advanced python\ese\image1.jpg", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)

            # Select image processing technique
            selected_technique = st.selectbox("Select Image Processing Technique", 
                                            ["Resize", "Grayscale", "Crop"])

            if selected_technique == "Resize":
                # Resize image
                new_width = st.number_input("Enter Width", value=image.width)
                new_height = st.number_input("Enter Height", value=image.height)
                resized_image = resize_image(image, new_width, new_height)
                st.image(resized_image, caption="Resized Image", use_column_width=True)

            elif selected_technique == "Grayscale":
                # Convert image to grayscale
                grayscale_image = grayscale_image(image)
                st.image(grayscale_image, caption="Grayscale Image", use_column_width=True)

            elif selected_technique == "Crop":
                # Crop image
                image_width, image_height = image.size
                left = st.slider("Left", 0, image_width, 0)
                top = st.slider("Top", 0, image_height, 0)
                right = st.slider("Right", 0, image_width, image_width)
                bottom = st.slider("Bottom", 0, image_height, image_height)
                cropped_image = crop_image(image, left, top, right, bottom)
                st.image(cropped_image, caption="Cropped Image", use_column_width=True)

    if __name__ == "__main__":
        main()



