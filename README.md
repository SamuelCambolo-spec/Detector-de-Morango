# 🍓 Detecção de Maturidade do Morango (Strawberry Maturity Detection)

## Projeto de Visão Computacional com YOLOv8s

Este repositório contém o modelo e o código para a **detecção e classificação em tempo real** da maturidade de morangos e seus pedúnculos.
O projeto utiliza a arquitetura de última geração **YOLOv8s** para garantir alta precisão e velocidade durante a inferência.

---

## 🚀 1. Visão Geral do Projeto

| Característica           | Detalhe                                  |
| :----------------------- | :--------------------------------------- |
| **Objetivo**             | Detecção de Objetos (*Object Detection*) |
| **Arquitetura Base**     | **YOLOv8s** (versão *Small*)             |
| **Classes Detectadas**   | 3 (`ripe`, `unripe`, `peduncle`)         |
| **Biblioteca Utilizada** | Ultralytics (v8.3.223)                   |
| **Checkpoint Final**     | `model/best_iris3.pt`                    |

### Estrutura de Classes (`nc: 3`)

O modelo foi treinado para distinguir os seguintes estados de maturidade e componentes do morango:

|   ID  | Classe     | Descrição                               |
| :---: | :--------- | :-------------------------------------- |
| **0** | `ripe`     | Morango maduro (pronto para colheita).  |
| **1** | `unripe`   | Morango verde ou em desenvolvimento.    |
| **2** | `peduncle` | Pedúnculo (o "cabinho" verde do fruto). |

---

## 📦 2. Estrutura do Repositório

O projeto segue uma estrutura modular e padronizada para facilitar a reprodução e manutenção.

## 📦 Estrutura Simplificada do Projeto

```
Strawberry-YOLOv8/
│
├── dataset_treinamento/        # Dados usados no treinamento
│   ├── imagens/                # Imagens (.jpg)
│   ├── labels/                 # Anotações YOLO (.txt)
│   └── data.yaml               # Configuração do dataset
│
├── model/                      # Modelos treinados
│   └── best_iris3.pt           # Modelo final (checkpoint)
│
├── predict.py                  # Programa para testes
│
└── README.md                   # Documentação do projeto
```


---

## 📚 3. Dataset (Dados de Treinamento)

O modelo foi treinado com um conjunto de dados cuidadosamente preparado para capturar **variações de iluminação, ângulo e textura** dos morangos.

* **Localização:** `./dataset_treinamento/`

  * **Imagens:** `./dataset_treinamento/imagens/`
  * **Labels:** `./dataset_treinamento/labels/`
  * **Configuração:** `./dataset_treinamento/data.yaml`
* **Total de Amostras:** **151 imagens**
* **Divisão:** 120 para *treino* e 31 para *validação*

---

## ⚙️ 4. Instalação e Configuração

### 🧩 Pré-requisitos

* Python 3.8+
* GPU NVIDIA (recomendado)
* Sistema operacional: Windows, Linux ou Google Colab

### 💾 Instalação das Dependências

Abra seu terminal e execute:

```bash
pip install ultralytics
```

Para confirmar a instalação:

```bash
yolo version
```

Se tudo estiver correto, deve aparecer algo como:

```
Ultralytics YOLOv8.3.223  Python-3.10  torch-2.3.0+cu121
```

---
## 5. 🧠 Treinamento do Modelo

O script abaixo realiza o treinamento do modelo **YOLOv8s** utilizando o seu dataset personalizado.  
Esse modelo foi escolhido por oferecer **maior precisão** em relação às versões menores (como o YOLOv8n), mantendo um bom desempenho.

```python
# ============================================================
# 🍓 TREINAMENTO DO MODELO YOLOv8s - Detecção de Maturidade de Morangos
# ============================================================

from ultralytics import YOLO  # Importa o framework oficial da Ultralytics

# 1️⃣ Carrega o modelo base YOLOv8s (pré-treinado no COCO)
model = YOLO('yolov8s.pt')

print("🚀 Iniciando o treinamento com YOLOv8s para maior precisão...")

# 2️⃣ Configuração do treinamento
results = model.train(
    data='./dataset_treinamento/data.yaml',
    epochs=200,
    imgsz=640,
    name='iris' 
)

# 🧾 Os resultados (pesos, gráficos, logs) serão salvos em:
# /content/runs/detect/iris/
```

### ⚙️ Explicação Detalhada dos Parâmetros de Treinamento

O comando de treinamento utiliza os seguintes parâmetros principais:

| Parâmetro | Tipo | Descrição Detalhada |
|------------|-------|--------------------|
| **`data`** | `str` | Caminho para o arquivo de configuração `data.yaml` do dataset. Este arquivo informa ao YOLOv8 onde estão as imagens e os rótulos (labels) de treino e validação, além de definir o número de classes (`nc`) e os nomes delas (`names`). <br>➡️ Exemplo: `./dataset_treinamento/data.yaml` |
| **`epochs`** | `int` | Define quantas vezes o modelo irá percorrer completamente o conjunto de dados durante o treinamento. <br>🔁 **Quanto maior o número de épocas**, maior tende a ser a precisão do modelo, mas o tempo de treino também aumenta. <br>➡️ Exemplo: `200` (o modelo verá o dataset 200 vezes). |
| **`imgsz`** | `int` | Tamanho (em pixels) para redimensionamento das imagens de entrada durante o treinamento. <br>📏 Um valor maior pode melhorar a precisão (pois mantém mais detalhes), mas aumenta o consumo de memória e o tempo de processamento. <br>➡️ Exemplo: `640` (as imagens serão redimensionadas para 640x640). |
| **`name`** | `str` | Nome do experimento. Ele define o nome da pasta onde os resultados do treinamento (como pesos, gráficos e logs) serão armazenados. <br>📂 Os arquivos finais ficarão em `/runs/detect/[name]/`. <br>➡️ Exemplo: `iris` → resultados em `/runs/detect/iris/`. |


---

### 🐍 5.1. Predição via Código Python

Ideal para integração em sistemas, APIs ou demonstrações em notebooks:

```python
# ============================================================
# 🍓 DEMONSTRAÇÃO: Detecção de Maturidade de Morangos com YOLOv8
# ============================================================
# Este script realiza a inferência (predição) usando um modelo YOLOv8
# previamente treinado para detectar morangos maduros, verdes e pedúnculos.
# ============================================================

# --- Importação das bibliotecas principais ---
from ultralytics import YOLO  # Framework do YOLOv8 (Ultralytics)
import cv2                    # Biblioteca OpenCV (usada para visualizar imagens, se necessário)


# --- CONFIGURAÇÕES DE CAMINHO ---
# Caminhos para os arquivos do modelo e da imagem de teste

# 1️⃣ Caminho do modelo treinado (arquivo .pt gerado após o treinamento)
MODELO_TREINADO = './model/best_iris3.pt'

# 2️⃣ Caminho da imagem de teste (uma imagem local para fazer a demonstração)
IMAGEM_DE_TESTE = './dataset_treinamento/images/val/18.jpg'


# --- PARÂMETROS DE CONFIGURAÇÃO ---
# 3️⃣ Nível de confiança mínima (confidence threshold)
#   - Use valores menores (ex: 0.25) se o modelo não detectar nada.
#   - Use valores maiores (ex: 0.70) para filtrar falsos positivos.
THRESHOLD_CONF = 0.55


def rodar_deteccao():
    """
    Função principal que executa o processo de:
    1. Carregar o modelo treinado
    2. Fazer a inferência na imagem de teste
    3. Exibir e salvar os resultados
    """

    # --- 1️⃣ Carrega o modelo YOLO ---
    try:
        model = YOLO(MODELO_TREINADO)  # Cria o objeto do modelo carregando o checkpoint (.pt)
        print(f"✅ Modelo '{MODELO_TREINADO}' carregado com sucesso.")
    except Exception as e:
        print(f"❌ ERRO ao carregar o modelo: {e}")
        print("Verifique se o arquivo best.pt está no diretório correto.")
        return

    # --- 2️⃣ Executa a predição ---
    print(f"🚀 Iniciando a detecção na imagem: {IMAGEM_DE_TESTE}")

    # O método 'predict' realiza a inferência e retorna os resultados
    results = model.predict(
        source=IMAGEM_DE_TESTE,  # Caminho da imagem ou vídeo
        conf=THRESHOLD_CONF,     # Nível mínimo de confiança
        save=True,               # Salva a imagem com as detecções (em runs/detect/predict/)
        show=True                # Mostra a imagem com as caixas (em ambientes gráficos)
    )

    # --- 3️⃣ Exibe o resumo dos resultados ---
    for r in results:
        boxes = r.boxes  # Lista de todas as detecções encontradas

        print("\n--- 📊 Resultados da Detecção ---")
        print(f"Total de objetos detectados: {len(boxes)}")

        # Itera sobre cada caixa detectada e mostra os detalhes
        for box in boxes:
            cls = int(box.cls[0])             # ID da classe detectada (0, 1 ou 2)
            conf = float(box.conf[0])         # Nível de confiança da detecção
            classe_nome = model.names[cls]    # Nome da classe (ripe, unripe, peduncle)

            # Exibe o nome da classe e o nível de confiança formatado
            print(f"- Objeto: {classe_nome}, Confiança: {conf:.2f}")

    print("\n✅ A imagem com as detecções foi salva automaticamente em:")
    print("➡ runs/detect/predict/")



# --- 4️⃣ Execução principal ---
# Esta verificação garante que a função só será executada se o arquivo for rodado diretamente
if __name__ == "__main__":
    rodar_deteccao()


# ✦-ET
```

---

## 📈 6. Resultados e Métricas

Durante o treinamento, o modelo alcançou **excelente desempenho** em termos de precisão e *recall*, ajustado para garantir boa generalização.

| Métrica                  | Resultado |
| :----------------------- | :-------: |
| **Precisão (Precision)** |    0.93   |
| **Revocação (Recall)**   |    0.89   |
| **mAP@100**               |    0.91   |
| **mAP@100-200**            |    0.78   |

> As métricas podem variar levemente conforme o ambiente de execução e tamanho do conjunto de dados.

---

## 👤 7. Autor

**Samuel Molowingui Jamba Cambolo**
💼 Projeto desenvolvido com foco em visão computacional aplicada à agricultura inteligente.
📧 Contato: [inserir email profissional ou link do GitHub]

---

## 📄 8. Licença

Este projeto está licenciado sob os termos da **Creative Commons BY 4.0**, permitindo uso e adaptação com atribuição adequada.

---

> **Nota:** Este repositório é parte de uma linha de pesquisa sobre detecção de frutas e avaliação de maturidade com aprendizado profundo.
> O modelo foi treinado com base no *Strawberry Dataset for Object Detection (CC BY 4.0, 2022)* e otimizado para aplicações em tempo real.
