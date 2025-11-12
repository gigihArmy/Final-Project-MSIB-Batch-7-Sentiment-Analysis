import libraries as lib
from collections import Counter

def run():
    lib.st.title("Sentiment Classification")
    lib.st.write("Proses Klasifikasi Sentimen Menggunakan SVM")
    
    uploaded_file = lib.st.file_uploader('Upload Dataset CSV', type='csv')
    
    if uploaded_file:
        lib.st.success('File telah terupload')
        df = lib.pd.read_csv(uploaded_file)
        lib.st.write('Raw Dataset')
        lib.st.write(df.head())
        lib.st.write(df.shape)
        
        df = df[['text', 'sentiment']]
        df = df.drop_duplicates()
        
        #Mengambil list stopwords
        stopwords_list = lib.stopwords.words('english')
        
        #custom kamus normalisasi
        norm_dict = {
            "u": "you",
            "r": "are",
            "nonsence": "nonsense",
            "distrube": "disturbe",
            "comming": "coming",
            "juss": "just",
            "shtdown": "shutdown",
            "soo": "so",
            "eng": "england",
            "au": "australia",
            "goin": "going",
            "insanedefaults": "insane defaults",
            "lmao": "laughing my ass off",
            "smh": "shake my head",
            "nyc": "new york city",
            "qood": "good",
            "dosen": "does not",
            "goodmorning": "good morning",
            "lol": "laughing out loud",
            "whoeva": "who ever",
}
        
        # Inisialisasi stemmer
        stemmer = lib.PorterStemmer()
        
        #Fungsi ekspansi grammar
        def expand_contractions(text):
            # Ganti karakter backtick (`) dengan apostrof standar (')
            text = lib.re.sub(r"[`´’]", "'", text)
            
            # Ekspansi kontraksi (don't=do not, i've=i have)
            expanded_text = lib.contractions.fix(text)
            return expanded_text

        #Fungsi Cleaning
        def clean_text(text):
            #casefolding
            text = text.lower()
    
            # Regex untuk menghapus URL dengan pola http atau https
            text = lib.re.sub(r"http[s]?://\S+", "", text)
            
            #remove stopwords
            remove_stopwords = [word for word in text.split() if word not in stopwords_list]
    
            #remove punc
            text = text.translate(str.maketrans('', '', lib.string.punctuation))
    
            #remove repeat char like "aaaa" to "a"
            text = lib.re.sub(r'(.)\1{2,}', r'\1', text)
            
            #remove single char like "x", "y"
            text = lib.re.sub(r'\b[a-zA-Z]\b', '', text)
            
            #remove digits
            text = lib.re.sub(r'\d', '', text)
    
            #remove whitespace
            text = lib.re.sub(r'\s+', ' ', text)
    
            return text
    
        #Fungsi Normalisasi kata
        def normalize_text(text, norm_dict):
            text = lib.re.sub(r'\s+', ' ', text)
            words = text.split()
            normalized_words = [norm_dict.get(word.lower(), word) for word in words]
            return ' '.join(normalized_words)
        
        #Fungsi stemming
        def stemming(text):
            stemmed = text.split()
            stemmed = [stemmer.stem(word) for word in stemmed if not word in lib.stopwords.words('english')]
            stemmed = " ".join(stemmed)
            return stemmed
        
        #Fungsi menghitung kata unik dalam X_train
        def unique_words(X_train):
            all_words = []
            for text in X_train:
                words = text.split()
                all_words.extend(words)
            word_counts = Counter(all_words)
            return len(word_counts)
        
        #Apply fungsi
        df['cleaned_text'] = df['text'].apply(expand_contractions)
        df['cleaned_text'] = df['cleaned_text'].apply(clean_text)
        
        lib.st.write("")
        lib.st.write("")
        lib.st.write('Clean Dataset')
        lib.st.write(df['cleaned_text'].head())
        lib.st.write(df.shape)
        
        df['cleaned_text'] = df['cleaned_text'].apply(normalize_text, args=(norm_dict,))
        df['stemmed'] = df['cleaned_text'].apply(stemming)
        
        lib.st.write("")
        lib.st.write("")
        lib.st.write('Normalized and Stemmed Dataset')
        lib.st.write(df['stemmed'].head())
        lib.st.write(df.shape)
        
        #Split Dataset
        X = df['stemmed']
        y = df['sentiment']

        X_train, X_test, y_train, y_test = lib.train_test_split(X, y, test_size=0.2, random_state=42)
        
        unique_word_count = unique_words(X_train)
        lib.st.write('Total Unique words in X_train: ', unique_word_count)
        
        unique_word_counts = X_train.apply(lambda x: len(set(x.split())))
        fig, ax = lib.plt.subplots(figsize=(8, 6))
        lib.sns.histplot(unique_word_counts, kde=True)
        lib.plt.title('Distribusi kata unik setiap baris pada X_train')
        lib.plt.xlabel('Jumlah kata-kata unik')
        lib.plt.ylabel('Frekuensi')
        lib.st.pyplot(fig)
        
        #Feature extraction
        vectorizer = lib.TfidfVectorizer(
        stop_words = stopwords_list,
        ngram_range = (1, 1),
        max_features = 50000
        )

        vectorizer.fit(X_train)
        X_train_vectorized = vectorizer.transform(X_train)
        X_test_vectorized = vectorizer.transform(X_test)
        
        lib.st.write("")
        lib.st.write("")
        lib.st.write('Model SVM')
        
        model = lib.SVC(kernel='rbf', C=10, gamma=0.1)
        model.fit(X_train_vectorized, y_train)
        
        y_pred = model.predict(X_test_vectorized)
        accuracy = lib.accuracy_score(y_test, y_pred)
        report_dict = lib.classification_report(y_test, y_pred, output_dict=True)
        report_df = lib.pd.DataFrame(report_dict).transpose()
        
        lib.st.text(f'Akurasi Model SVM: {accuracy * 100:.2f}%')
        lib.st.text('neutral: 0 \npositive: 1\nnegative: 2')
        lib.st.dataframe(report_df.style.format(precision=2))
        
        lib.st.write("")
        fig, ax = lib.plt.subplots(figsize=(8,6))
        lib.plt.title('Heatmap Hasil Prediksi Klasifikasi Model SVM')
        lib.sns.heatmap(lib.confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        lib.plt.xlabel('Predicted')
        lib.plt.ylabel('Actual')
        lib.st.pyplot(fig)
        
        #save model
        with open('sentiment_classification_svm.pkl','wb') as f:
            lib.pickle.dump(model,f)
    
        with open('vectorize.pkl','wb') as f:
            lib.pickle.dump(vectorizer,f)
            
        #load model
        with open('sentiment_classification_svm.pkl','rb') as f:
            loaded_model = lib.pickle.load(f)

        with open('vectorize.pkl','rb') as f:
            loaded_vectorizer = lib.pickle.load(f)
        
        lib.st.write("")
        lib.st.write("")
        lib.st.write('Predict Whole Text (X)')
        lib.st.text("Prediksi Keseluruhan Dataset Teks tanpa split data train")
        # Transform X
        X_vectorized = loaded_vectorizer.transform(X)

        # Predict
        pred = loaded_model.predict(X_vectorized)

        df['sentiment_prediction'] = pred
        
        # Mengembalikan nilai numerik menjadi string
        df['sentiment'] = df['sentiment'].replace(
            {
                0: 'neutral',
                1: 'positive',
                2: 'negative'
            }
        )
        
        df = df[['text','sentiment_prediction']]
        lib.st.write(df[['text','sentiment_prediction']].head())
        
        data_counts = df['sentiment_prediction'].value_counts()

        fig, ax = lib.plt.subplots(figsize=(8, 6))
        ax = lib.sns.barplot(x = data_counts.index, y = data_counts.values, palette='pastel')
        lib.plt.title('Perbandingan Hasil Prediksi Kelas Sentimen Seluruh Data Teks')
        lib.plt.xlabel('Kelas Sentimen')
        lib.plt.ylabel('Jumlah Data')

        for i, count in enumerate(data_counts.values):
            ax.text(i, count+0.10, str(count), ha='center', va='bottom')

        lib.st.pyplot(fig)
        
        lib.st.write("")
        lib.st.write("")
        lib.st.write('Split Sentiment Class')
        
        #positive class
        lib.st.write("")
        lib.st.text('Positive Class')
        positive_sentiment = df[df['sentiment_prediction'] == 'positive']
        lib.st.write(positive_sentiment.head())
        lib.st.write(positive_sentiment.shape)

        csv_pos = positive_sentiment.to_csv(index=False)

        # Menyediakan tombol unduhan
        lib.st.download_button(
            label="Download hasil klasifikasi positif",
            data=csv_pos,
            file_name='positive_sentiment.csv',
            mime='text/csv'
        )
        
   
        #negative class
        lib.st.write("")
        lib.st.text('Negative Class')
        negative_sentiment = df[df['sentiment_prediction'] == 'negative']
        lib.st.write(negative_sentiment.head())
        lib.st.write(negative_sentiment.shape)
        
        csv_neg = negative_sentiment.to_csv(index=False)

        # Menyediakan tombol unduhan
        lib.st.download_button(
            label="Download hasil klasifikasi negatif",
            data=csv_neg,
            file_name='negative_sentiment.csv',
            mime='text/csv'
        )
        
        #neutral class
        lib.st.write("")
        lib.st.text('Neutral Class')
        neutral_sentiment = df[df['sentiment_prediction'] == 'neutral']
        lib.st.write(neutral_sentiment.head())
        lib.st.write(neutral_sentiment.shape)
        
        csv_neu = neutral_sentiment.to_csv(index=False)

        # Menyediakan tombol unduhan
        lib.st.download_button(
            label="Download hasil klasifikasi netral",
            data=csv_neu,
            file_name='neutral_sentiment.csv',
            mime='text/csv'
        )
        
     
        
        
        
        
        
        
        
