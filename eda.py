import libraries as lib

def run():
    lib.st.title("Exploration Data Analysis")
    lib.st.write("Eksplorasi Dataset Media Sosial")
    
    uploaded_file = lib.st.file_uploader('Upload Dataset CSV', type='csv')
    
    if uploaded_file:
        lib.st.success('File telah terupload')
        df = lib.pd.read_csv(uploaded_file)
        lib.st.write('Dataframe')
        lib.st.write(df.head())
        lib.st.write(df.shape)
        df['Platform'] = df['Platform'].str.strip().str.lower()
        
        lib.st.write("")
        lib.st.write("")
        lib.st.write("")
        lib.st.write('Distribusi Data Bar Chart')
        col = lib.st.selectbox('Pilih kolom', options=df.columns)
        
        if col:
            data_counts = df[col].value_counts()
                
            if len(data_counts) > 10:
                lib.st.warning(f'Kolom {col} memiliki index terlalu banyak. Coba gunakan grouping untuk melihat lebih lanjut.')
            else:
                fig, ax = lib.plt.subplots(figsize=(8, 6))
                ax = lib.sns.barplot(x=data_counts.index, y=data_counts.values, palette='pastel')
                lib.plt.title(f'Distribusi Data Kolom {col}')
                lib.plt.xlabel(col)
                lib.plt.ylabel('Jumlah Data')

                for i, count in enumerate(data_counts.values):
                    ax.text(i, count + 0.10, str(count), ha='center', va='bottom')

                lib.st.pyplot(fig)
        
        lib.st.write("")
        lib.st.write("")
        lib.st.write("")
        lib.st.write('Grouping Data Heat Map')
        col_satu = lib.st.selectbox('Pilih kolom Satu', options=df.columns, key='heatmap_col1')
        col_dua = lib.st.selectbox('Pilih kolom Dua', options=df.columns, key='heatmap_col2')
        
        num_unique_satu = df[col_satu].nunique()
        num_unique_dua = df[col_dua].nunique()
        
        if col_satu == col_dua:
            lib.st.warning('Kolom tidak boleh sama')
        elif num_unique_satu > 100 or num_unique_dua > 100:
            lib.st.warning('Kolom yang dipilih memiliki index terlalu banyak')
        else:
            map_vis = df.groupby([col_satu, col_dua]).size().unstack(fill_value=0)
                
            fig, ax = lib.plt.subplots(figsize=(8, 6))
            lib.sns.heatmap(map_vis, annot=True, cmap='coolwarm', fmt='d', ax=ax)
            lib.plt.title(f'Distribusi {col_dua} Berdasarkan {col_satu}')
            lib.st.pyplot(fig)
        
        lib.st.write("")
        lib.st.write("")
        lib.st.write("")
        lib.st.write("Word Cloud")
        lib.st.text("Ukuran kata dalam wordcloud biasanya berbanding lurus dengan frekuensi kemunculannya, semakin sering kata muncul maka besar tampilannya.")
        
        #Dataset dibersihkan terlebih dahulu
        def expand_contractions(text):
            # Ganti karakter backtick (`) dengan apostrof standar (')
            text = lib.re.sub(r"[`´’]", "'", text)
            
            # Ekspansi kontraksi (don't=do not, i've=i have)
            expanded_text = lib.contractions.fix(text)
            return expanded_text

        #cleaning function
        def clean_text(text):
            #casefolding
            text = text.lower()
    
            # Regex untuk menghapus URL dengan pola http atau https
            text = lib.re.sub(r"http[s]?://\S+", "", text)
    
            #remove punc
            text = text.translate(str.maketrans('', '', lib.string.punctuation))
    
            #remove repeat char like "aaaa" to "a"
            text = lib.re.sub(r'(.)\1{2,}', r'\1', text)
    
            #remove whitespace
            text = lib.re.sub(r'\s+', ' ', text)
    
            return text
    
        df['text'] = df['text'].apply(expand_contractions)
        df['text'] = df['text'].apply(clean_text)
            
        texts = df['text']
            
        all_text = ' '.join(texts)
        fig, ax = lib.plt.subplots(figsize=(8, 6))
        wordcloud = lib.WordCloud(width=800, height=800, background_color='white').generate(all_text)

        lib.plt.imshow(wordcloud, interpolation='bilinear')
        lib.plt.axis('off')
        lib.plt.title('Word Cloud Data Teks')
        lib.st.pyplot(fig)
        