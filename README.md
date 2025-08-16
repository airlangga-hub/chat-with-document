# 🖥️ Web App Overview
This is a simple **RAG Web App** where the user can have a *conversation with their document*.\
In the example below, I used [Apple Inc.'s 2025 Q2 10-Q document.](https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/b986f1de-d226-4e8e-9304-29a8458440ec.pdf)\
![video](video.mp4)

# 🦙 The LLM
My LLM of choice was **llama3-70b-8192** via [Groq](https://console.groq.com/docs/models).\
The reason for choosing this model is because it has MMLU score of 86% accuracy, making it suitable for RAG task.

# Embedding Model
The embedding model used is the `sentence-transformers/all-mpnet-base-v2`.\
The reason is because it is `light` and `fast`.

# Vector DB
The Vector DB used is FAISS.\
The reason is because it efficiently stores and retrieves high-dimensional embeddings, allowing fast similarity search between user queries and document chunks.

# Update Knowledge Base
To update the knowledge base, simply upload a new PDF file. The system will process it, create updated document chunks, and rebuild the vector store with the latest content, ensuring the chatbot answers based on the most current document.

# Weakness of this RAG App
It doesn't use persistent data base so a user needs to upload a document every time they need to query the LLM about the document. This is because it is a small project and doesn't need to use persistent database.
