import fitz

input_pdf = "catalogue-uv/uv_catalogue.pdf"
output_pdf = "catalogue-uv/uv_catalogue_extracted.pdf"

doc = fitz.open(input_pdf)
new_doc = fitz.open()

# Pages 36 à 194 (toutes les UV du catalogue)
for page_num in range(35, 194):
    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

new_doc.save(output_pdf)
new_doc.close()
doc.close()

print(f"Extrait {194-36+1} pages vers {output_pdf}")
