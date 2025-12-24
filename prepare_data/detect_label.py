import pickle

with open("./data/preprocessed_chunks/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print(le.classes_)  # danh sách tên class gốc theo đúng index 0..n-1