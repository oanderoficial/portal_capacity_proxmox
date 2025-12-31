from app.dashboard import create_app

app = create_app()

if __name__ == "__main__":
    app.run(
    debug=True,
    host="0.0.0.0",   # permite acesso externo ao portal
    port=8050         # porta do dashboard
)
