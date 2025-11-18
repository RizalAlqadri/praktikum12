from utils_facenet import embed_from_path, cosine_similarity

img1 = "data/val/rizal/rizal1.jpeg"
img2 = "data/val/rizal/rizal2.jpeg"

emb1 = embed_from_path(img1)
emb2 = embed_from_path(img2)

if emb1 is None or emb2 is None:
    print("❌ Wajah tidak terdeteksi.")
else:
    sim = cosine_similarity(emb1, emb2)
    print("Similarity:", sim)
    print("Match?" , "YA" if sim >= 0.85 else "TIDAK")