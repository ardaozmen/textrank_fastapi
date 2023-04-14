## Keyword Extraction using TextRank algorithm with FastApi



### Overview

- This project is an implementation of the TextRank algorithm for keyword extraction and its deployment using FastAPI. The algorithm uses a graph-based approach to identify the most important keywords in a given text.


> :warning: **Note**: This algorithm does not actually perform summarization, as it selects top sentences without regard for their position or order in the original text.


### Installation

1. Clone the repository using the following command:
```prompt
git clone https://github.com/ardaozmen/textrank_fastapi.git
```


2. Navigate to the project directory:
```prompt
cd textrank_fastapi-master
```

### Usage

To start the FastAPI server, run the following command:
```prompt
uvicorn app:app --reload
```
This will start the server on http://localhost:8000.


If it will not start that port:
```prompt
uvicorn app:app --reload --port 8001
```

### Contributing

If you find any issues or want to contribute to this project, please feel free to open a pull request or an issue.