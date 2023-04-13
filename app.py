# Word Cloud
import base64
from io import BytesIO
from starlette.routing import request_response
from wordcloud import WordCloud, STOPWORDS

# FastAPI
import uvicorn
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# TextRank Class
from core.textrank import TextRank4Sentences

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# nltk.download('punkt') # download this

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def home(request: Request):
    sentence_count = 0
    summary = ""
    if request.method == "POST":
        form = await request.form()
        if form["message"] and form["sentence_count"]:
            sentence_count = form["sentence_count"]
            text_str = form["message"]
            sentence_count = int(sentence_count)

            tr4sh = TextRank4Sentences()
            tr4sh.analyze(text_str)
            summary = tr4sh.get_top_sentences(sentence_count)
            listToStr = ' '.join([str(elem) for elem in summary])
            word_cloud = wordcloud(listToStr)
                        
    return templates.TemplateResponse("index.html", {"request": request, "summary": listToStr, "wordcloud": word_cloud})


def wordcloud(text_str):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(text_str).to_image()
    img = BytesIO()
    wordcloud.save(img, "PNG")
    img.seek(0)
    img_b64 = base64.b64encode(img.getvalue()).decode()
    return img_b64


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
