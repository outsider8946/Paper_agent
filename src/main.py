from rag import RAG

def main():
    rag_system = RAG(path2pdf='pdf/attention_pdf.pdf', model_name='qwen3:4b', debug=False)

    while True:
        user_query = input('Your question: ')
        print(f'RAG answer:\n{rag_system(user_query)}')

if __name__ == '__main__':
    main()