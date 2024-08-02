
from torch.utils.data import DataLoader,TensorDataset
import torch
from transformers import BertTokenizer,BertConfig,BertForQuestionAnswering,GPT2Tokenizer,GPT2LMHeadModel
from pymongo import MongoClient


def get_mongodb():
    # MongoDB bağlantı bilgilerini döndürecek şekilde tanımlanmalıdır.
    return 'mongodb://localhost:27017/', 'yeniDatabase', 'train'

def get_input_texts():
    # MongoDB bağlantı bilgilerini alma
    mongo_url, db_name, collection_name = get_mongodb()

    # MongoDB'ye bağlanma
    client = MongoClient(mongo_url)
    db = client[db_name]
    collection = db[collection_name]

    # Sorguyu tanımlama
    query = {"Prompt": {"$exists": True}}

    # Sorguyu çalıştırma ve dökümanları çekme
    cursor = collection.find(query, {"Prompt": 1, "Response":1, "_id": 0})  # 'input_text' alanını almak için "_id": 0 ekleyin

    # Cursor'ı döküman listesine dönüştürme
    input_texts_from_db = list(cursor)

    # Input text'leri döndürme
    return input_texts_from_db

def preapare_data(data,tokenizer,max_length=100):
    input_texts= [d['Propmt'] for d in data]
    output_texts= [d['Response'] for d in data]

    inputs=tokenizer.batch_encode_plus(
        input_texts,padding=True,truncation=True,max_length=max_length, return_tensors='pt'
    )
    outputs=tokenizer.batch_encode_plus(
        output_texts,padding=True,truncation=True,max_length=max_length, return_tensors='pt'
    )
    return inputs,outputs

train_data="C:\\newProjects\\train_Egitim\\merged_train.parquet"
test_data="C:\\newProjects\\test_Egitim\\merged_train.parquet"
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
train_inputs, train_outputs= preapare_data(train_data, tokenizer)
test_inputs, test_outputs= preapare_data(test_data, tokenizer)

class Model:
    
    def __init__(self,learning_rate= 1e-4,epochs=3,batch_size=100):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.batch_size=batch_size


    def train(self,model_path:str,encoded_inputs):
        print(f"Learning rate:{self.learning_rate}")
        print(f"Epochs:{self.epochs}")
        print(f"batch size: {self.batch_size}")

        #model yolunu ekleme
        model,tokenizer =self.load_model(model_path)
        #optimize etme 
        optimizer=torch.optim.Adam(model.parameters(),lr=self.learning_rate)

        #hdef değerlerle karşılaştırma yapabilmek için ve doğruluğu ölçmek için

        # Assuming you have tokenized input texts and labels
        #ınput tensorları
        #attetion mask bert dilinde modelin sadece gerçek tokenler üzerinde çalışmasını sağlar.
        input_ids = encoded_inputs['input_ids'] # Replace with your tokenized input texts
        attention_masks = encoded_inputs['attention_mask']
        labels = torch.tensor([1]*len(input_ids))

        #create a tensordataset 
        train_dataset=TensorDataset(train_inputs['input_ids'],train_outputs['input_ids'],labels)
        train_loader=DataLoader(train_dataset,batch_size=8, shuffle=True)
        # Create a data loader

        
        model.train()
        #Iterate over the epochs
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in data_loader:
            
                
                batch_input_ids, batch_attention_mask, batch_labels=batch
                #reset gradients
                optimizer.zero_grad()

                #forward pass 
                outputs = model(input_ids= batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
                loss = outputs.loss
                #backward pass 
                loss.backward()
                #update optimizer 
                optimizer.step()
                #accumulate total loss
                total_loss += loss.item()
        #calculate average loss
        average_loss = total_loss / len(encoded_inputs)
        #print the loss for current epoch
        print(f"Epoch {epoch+1} - Loss: {average_loss:.4f}")

    #tüm bu verileri tutan bir "batch_of_attention_masks" verisini tanımlamam gerek

    def load_model(self,model_path: str,do_lower_case=False):
        config = BertConfig.from_pretrained(model_path + "C:\\gitProjects\\train_Egitim")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
        return model, tokenizer

    def generate_response(input_text):
        inputs=tokenizer.encode(input_text,return_tensors='pt')
        outputs=model.generate(inputs,max_length=100,num_return_sequences=1)
        response= tokenizer.decode(outputs[0],skip_special_tokens=True) #buraya modelin tanımlaması yazılmalı 
        return response

#veri tabanından input textlerini al
input_texts_from_db= get_input_texts()
#encode etmek için gerekli olan bilgiler 
input_texts=[doc["Prompt"] for doc in input_texts_from_db ]


#tokenizer ı yükle
tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    


# Tokenize the input texts
encoded_inputs = tokenizer.batch_encode_plus(
    input_texts,
    padding=True,
    truncation=True,
    max_length=100,
    return_attention_mask=True,
    return_tensors='pt'
)