from pyngrok import ngrok
ngrok.kill()
!streamlit run app.py &>/dev/null&
public_url = ngrok.connect(8501)
print("Streamlit URL:", public_url)
