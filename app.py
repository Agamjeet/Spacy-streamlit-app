#core packages
import streamlit as st


#NLP packages
import spacy
import spacy_streamlit
nlp=spacy.load("en_core_web_sm")
try:
    # Attempt to load the spaCy model
    nlp = spacy.load("en_core_web_sm")
    print("Model 'en_core_web_sm' is already downloaded.")
except OSError:
    print("Model 'en_core_web_sm' is not downloaded.")

def main():
    '''A simple NLP app with Spacy-STREAMLIT'''
    st.title("Spacy-Streamlit NLP APP")
    menu=["Home","NER"]
    choice=st.sidebar.selectbox('Menu',menu)
    
    if choice=="Home":
        st.subheader("Tokenization")
        raw_text=st.text_area("Your text","Enter your text here")
        docx=nlp(raw_text)
        if st.button('tokenize'):
            spacy_streamlit.visualize_tokens(docx,attrs=['text','pos_','dep_','ent_type_'])
        
    elif choice =="NER":
        st.subheader("Named Entity Recoginition")
        raw_text=st.text_area("Your text","Enter your text here")
        docx=nlp(raw_text)
       
        spacy_streamlit.visualize_ner(docx,labels=nlp.get_pipe("ner").labels)


if __name__=='__main__':
    main()