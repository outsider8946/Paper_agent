import logging
from rag import RAG
from omegaconf import OmegaConf

def main():
    logging.basicConfig(level=logging.INFO)
    config = OmegaConf.load('config.yaml')
    rag_system = RAG(path2pdf='pdf/attention_pdf.pdf', config=config)

    while True:
        user_query = input('Your question: ')
        user_query = f'{user_query} /no_think'
        print(f'RAG answer:\n{rag_system(user_query)}')

if __name__ == '__main__':
    main()