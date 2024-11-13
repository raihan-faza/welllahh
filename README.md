# welllahh



### AI model
```
- indonesian food image to its name:
https://drive.google.com/file/d/1N3rkjM9C_hbyiLzKmwdTzxYYHwYS-RDW/view?usp=sharing
- Food Recommendation System:
https://drive.google.com/uc?export=download&id=1US8wd4AsafsVGDl-HWcD0AzqTm-l-FTK
- Backup Vector Database (ChromaDB), yang udah di index pake teks: https://huggingface.co/datasets/MedRAG/textbooks, sebagian https://huggingface.co/datasets/MedRAG/pubmed 
https://drive.google.com/file/d/1S0nxpfC4ifrktLpoetCrNmTyTVSydsqM/view?usp=sharing

```

### Note
- Ekstrak dulu welllahh_chroma.zip 
- kalau mau pakai food nutrition detector, uncomment code berikut di settings.py (cukup lama load nya sekitar 3-5 menit): 
```
print("loading food image classification model....")
INDOFOOD_IMAGE_MODEL = tf.keras.models.load_model("best_model_86.keras")
print("selesai food image classification model....")
```
- pake python versi 3.11.0 aja bang (bisa pake pyenv), soalnya modelnya cuma bisa di tensorflow 2.15.0 .
- agak lama emang run servernya, karena parameter modelnya gede
- buat pakai chatbot harus pakai ProtonVPN, karena pakai hasil search DuckDuckGoSearch (diblokir di indo).
- buat pakai chatbot, anda harus masukkan GEMINI_API_KEY anda ke .env 
- 
