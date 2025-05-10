
# Cyberbullying Classifier

Bu proje, 2025-AIWEBX HACKATHON’da elde ettiğimiz 1.'lik ödülüyle taçlandı.

Sosyal medyada siber zorbalığı tespit etmek ve sınıflandırmak amacıyla geliştirdiğimiz bu çözüm, güçlü veri işleme teknikleri, özel tasarlanmış GRU-Attention modeli ve yapay zeka tabanlı derin öğrenme yaklaşımları sayesinde öne çıktı.

Bu başarı, karmaşık dil yapılarını anlamak, zorbalık türlerini hassasiyetle ayırt etmek ve çevrimiçi ortamı daha güvenli hale getirmek için atılan önemli bir adım oldu.

## 📌 Proje Tanıtımı

"Cyberbullying Classifier" sosyal medya paylaşımlarındaki siber zorbalık içeriklerini tespit edip sınıflandıran bir web uygulamasıdır. Flask ile hazırlanmış arayüz ve arka plandaki Keras/TensorFlow tabanlı GRU-Attention modeli sayesinde kullanıcıdan alınan Tweet metinlerini aşağıdaki alt sınıflara ayırır:

-   `age` (Yaş ile ilgili zorbalık)
    
-   `ethnicity` (Etnik kökenle ilgili zorbalık)
    
-   `gender` (Cinsiyetle ilgili zorbalık)
    
-   `not_cyberbullying` (Zorbalık içermeyen paylaşımlar)
    
-   `other_cyberbullying` (Diğer zorbalık türleri)
    
-   `religion` (Din ile ilgili zorbalık)
    

## ⚙️ Özellikler

-   **Kapsamlı Metin Temizleme:** Emoji, kısaltma, link/mention kaldırma, lemmatizasyon, kısa ve tekrarlı kelime filtreleme
    
-   **Önceden Eğitilmiş Embedding:** 200 boyutlu GloVe vektörleri
    
-   **GRU + Dikkat Mekanizması:** Önemli kelimelere odaklanan Attention katmanı
    
-   **Veri Dengelenmesi:** SMOTE ve sınıf ağırlıklarıyla azınlık sınıflarına destek
    
-   **Kolay Kullanım:** Basit REST API üzerinden JSON tabanlı tahmin
    
-   **Web Arayüzü:** Kullanıcı dostu form ve statik görsellerle sonuç sunumu
    

## 🖥️ Sistem Gereksinimleri

-   Python 3.7+
    
-   TensorFlow 2.x
    
-   Flask
    
-   imbalanced-learn
    
-   NLTK, demoji, contractions, langdetect, seaborn, matplotlib
    


## 🚀 Kurulum Adımları

1.  Depoyu klonlayın:
    
    ```bash
    git clone https://github.com/<kullanıcı-adınız>/Cyberbullying-Classifier.git
    cd Cyberbullying-Classifier
    
    ```
    
2.  Sanal ortam oluşturun ve aktif edin:
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate    # macOS/Linux
    venv\Scripts\activate      # Windows
    
    ```
    
3.  Gerekli paketleri yükleyin:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
4.  Emoji veritabanını indirin (ilk çalıştırma için internet gerektirir):
    
    ```python
    python - <<EOF
    import demoji; demoji.download_codes()
    EOF
    
    ```
    

## ▶️ Uygulamanın Çalıştırılması

```bash
python app.py
```

Tarayıcınızda `http://127.0.0.1:5000` adresini açarak ana sayfaya ulaşabilirsiniz.

-   **Tahmin Sayfası:** Tweet’inizi girip "Predict" butonuna tıklayın.
    
-   **Görseller Sayfası:** Modelin karışıklık matrisi ve F1 skor grafiklerini inceleyin.
    

## 📂 Proje Yapısı

```
├── app.py                     # Flask uygulaması
├── utils.py                   # Metin temizleme ve Attention katmanı
├── datasets/                  # Ham ve test verisetleri (dataset.csv, test.csv)
├── models/                    # Eğitilmiş modeller (.h5) ve tokenizer.pkl
├── static/                    # Görseller ve CSS/JS dosyaları
├── templates/                 # index.html ve charts.html
├── notebooks/                 # Model eğitim & analiz notebook'ları
└── requirements.txt           # Bağımlılıklar

```

## 📊 Model Detayları

-   **Mimari:** Embedding → Bidirectional GRU → Custom Attention → Dense katmanlar → Softmax
    
-   **Eğitim:** EarlyStopping, ModelCheckpoint, sınıf ağırlıkları
    
-   **Performans:** Test doğruluğu ~85%, alt sınıflarda F1 skorları detaylı olarak `charts.html` sayfasında.
    



----------

### Mahmut Kerem Erden - k.erden03@gmail.com
