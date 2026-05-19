# Urban Demand Oracle

Predicting NYC taxi demand by zone and hour — with a Claude-powered natural language analyst.

## Stack
- **ETL**: Python pipeline → PostgreSQL
- **ML**: XGBoost + SHAP explainability
- **API**: FastAPI (`/predict` + `/ask`)
- **AI**: Claude (Anthropic API) for natural language Q&A over the data
- **Dashboard**: Streamlit
- **Infrastructure**: Docker Compose

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/urban-demand-oracle
cd urban-demand-oracle
cp .env.example .env        # add your API keys
make up
```

## Architecture
_Diagram coming in final polish step._

## Author
Eric Planas — Senior Data Scientist
