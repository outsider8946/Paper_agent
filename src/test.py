from langchain_text_splitters import MarkdownHeaderTextSplitter

with open('output/attention_pdf.mmd', 'r', encoding='utf-8') as f:
            content = f.read()

headers_to_split_on = [
        ('#', 'Header 1'),
        ('##', 'Header 2'),
        ('###', 'Header 3')
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
splits = md_splitter.split_text(content)

for i, item in enumerate(splits):
    print(f'split {i}\n\n')
    print(item)

