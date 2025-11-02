from ultralytics import YOLO
import cv2

# --- CONFIGURAÇÕES DE CAMINHO ---
# 1. Caminho para o seu modelo treinado (best.pt)
MODELO_TREINADO = './model/best_iris3.pt' 

# 2. Caminho para a imagem que você deseja testar (sua imagem de demonstração)
IMAGEM_DE_TESTE = './dataset_treinamento/images/val/18.jpg' 

# --- PARÂMETROS ---
# 3. Confiança Mínima (Confidence Threshold):
# Diminua se o modelo não estiver detectando nada (ex: 0.25). 
# Aumente se houver muitas detecções erradas (ex: 0.70).
THRESHOLD_CONF = 0.55 


def rodar_deteccao():
    """Carrega o modelo e executa a detecção em uma imagem local."""
    
    # 1. Carrega o modelo
    try:
        model = YOLO(MODELO_TREINADO) 
        print(f"Modelo {MODELO_TREINADO} carregado com sucesso.")
    except Exception as e:
        print(f"ERRO ao carregar o modelo: {e}")
        print("Verifique se o arquivo best.pt está no diretório correto.")
        return

    # 2. Executa a predição (Inferencia)
    print(f"Iniciando a detecção na imagem: {IMAGEM_DE_TESTE}")
    results = model.predict(
        source=IMAGEM_DE_TESTE,
        conf=THRESHOLD_CONF,
        save=True,  # Salva a imagem resultante na pasta 'runs/detect/predict/'
        show=True   # Abre uma janela para mostrar a imagem com as caixas (se estiver em um ambiente gráfico)
    )

    # 3. Exibe o resumo do resultado
    for r in results:
        boxes = r.boxes
        print("\n--- Resultados da Detecção ---")
        print(f"Total de objetos detectados: {len(boxes)}")
        
        # Mostra as classes e confianças
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            classe_nome = model.names[cls]
            print(f"- Objeto: {classe_nome}, Confiança: {conf:.2f}")

    print(f"\nA imagem com as detecções foi salva na pasta 'runs/detect/predict/' no seu projeto.")

if __name__ == "__main__":
    rodar_deteccao()