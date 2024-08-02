import gradio as gr
from pymongo import MongoClient
import pandas as pd

# MongoDB'ye bağlanma
client = MongoClient('mongodb://localhost:27017/')
db = client['yeniDatabase']
collection = db['train']

# Model eğitme fonksiyonu (örnek)
def train_model(filtered_data):
    # Burada model eğitme işlemleri yapılır
    # Örneğin, sadece veri boyutunu döndüren sahte bir model eğitimi
    model_response = {
        'status': 'success',
        'message': 'Model trained successfully!',
        'accuracy': 0.95,  # Örnek doğruluk değeri
        'data_size': len(filtered_data)
    }
    return model_response

# Gradio uygulaması için fonksiyon
def train_model_gradio(title,keywords, subcategories, subheadings):
    # MongoDB'den ilgili verileri çekme
    query = {
        'title': {'$in': title},
        'category': {'$in': keywords.split(',')},
        'subcategory': {'$in': subcategories.split(',')},
        'subheadings': {'$in': subheadings.split(',')}
    }
    filtered_data = list(collection.find(query))

    # Model eğitme
    response = train_model(filtered_data)
    return response

# Gradio arayüzü
iface = gr.Interface(
    fn=train_model_gradio,
    inputs=[
        gr.Textbox(label="Title"),
        gr.Textbox(label="Keywords (comma-separated)"),
        gr.Textbox(label="Subcategories (comma-separated)"),
        gr.Textbox(label="Subheadings (comma-separated)")
    ],
    outputs="json",
    title="Model Training Interface",
    description="Enter the titles, categories, subcategories, and subheadings to filter the data and train the model."
)

if __name__ == "__main__":
    iface.launch()
