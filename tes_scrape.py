# #https://www.ncbi.nlm.nih.gov/books/NBK189860/
# from bs4 import BeautifulSoup
# import requests

# #npin.cdc.gov
# #ncbi.nlm.nih.gov
# # link = 'https://npin.cdc.gov/publication/nutritional-care-and-support-patients-tuberculosis'
# # link = "https://www.narayanahealth.org/blog/tuberculosis-diet-what-to-eat-and-what-to-avoid"

# link = "https://www.who.int/publications/i/item/9789241506410"
# try:
#     page = requests.get(link).text
# except requests.exceptions.RequestException as errh:
#     print(f"error: {errh}")

# doc = BeautifulSoup(page, "html.parser")
# print(doc.prettify())

# h2s = doc.find_all("h2")
# print(h2s)
# paragraphs = doc.find_all("p")
# text_content = "\n\n".join([para.get_text() for para in paragraphs])
# print(text_content)

link = "https://www.who.int/publications/i/item/9789241506410"
domain = link.split("/")[2]
print(domain)