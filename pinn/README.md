# ⚙️🔥 Estimativa de Temperatura Interna em Motores Elétricos via Physics‑Informed Neural Networks (PINNs)

> **Trabalho Final – Redes Neurais Artificiais e Deep Learning**  
> **Autor:** Thales Ferreira • **Validação prévia:** 09 / 06 • **Entrega final:** 16 / 06

---

## 📑 Índice
1. [🔮 Introdução](#-introdução)
2. [📚 Fundamentação Teórica](#-fundamentação-teórica)
3. [💾 Base de Dados & Domínio do Problema](#-base-de-dados--domínio-do-problema)
4. [🛠️ Pré‑processamento & Geração de Dados Sintéticos](#-préprocessamento--geração-de-dados-sintéticos)
5. [🏗️ Arquitetura PINN](#-arquitetura-pinn)
6. [🧪 Planejamento Experimental](#-planejamento-experimental)
7. [🔬 Experimentos & Resultados](#-experimentos--resultados)
8. [📊 Análise e Discussão](#-análise-e-discussão)
9. [🔚 Considerações Finais](#-considerações-finais)
10. [🚀 Reprodutibilidade](#-reprodutibilidade)
11. [📚 Referências](#-referências)

---

## 🔮 Introdução
Sobreaquecimento do enrolamento é uma das principais causas de falhas prematuras em motores elétricos industriais. **Sensores intrusivos** elevam custo e complexidade; logo, surge a necessidade de um **sensor virtual** que estime a temperatura interna a partir de variáveis fáceis de medir (corrente RMS, temperatura da carcaça). Neste trabalho proponho um modelo baseado em **Physics‑Informed Neural Networks (PINNs)** — redes que minimizam simultaneamente o erro nos dados disponíveis e o **resíduo da equação de calor 1‑D com fonte \(I^2R\)**.

### Hipótese
> Um PINN devidamente calibrado atingirá **MAE ≤ 5 °C** na estimativa da temperatura interna em regime de produção contínua.

---

## 📚 Fundamentação Teórica
### 2.1 Physics‑Informed Neural Networks
PINNs foram introduzidas por Raissi *et al.* [1], [2] para incorporar **leis de conservação** diretamente na função de perda, dispensando malhas numéricas clássicas. Surveys recentes destacam avanços em balanceamento adaptativo de perdas, generalização e aceleração [3], [4], [11].

### 2.2 Transferência de Calor em Máquinas Elétricas
Modelos térmicos de motores geralmente combinam elementos concentrados (Lumped‑Parameter Thermal Networks) e métodos numéricos [5]. Estudos recentes demonstraram que PINNs conseguem igualar ou superar esses modelos, mesmo com dados esparsos, em casos de motores de indução e PMSM [6]–[9].

### 2.3 Ferramentas de Implementação
Bibliotecas como **DeepXDE** [10] e **SciANN** [12] fornecem APIs de alto nível para PINNs, possibilitando o uso de autograd em TensorFlow/PyTorch e rápida prototipagem. Outras iniciativas, como **PINE** [4] e **A‑PINN survey** [11], exploram otimização evolutiva e meta‑aprendizado para contornar rigidez numérica.

> **Síntese:** A literatura confirma a pertinência dos PINNs para problemas térmicos em motores, mas poucos estudos tratam de **linhas de produção contínua** — lacuna que este projeto aborda.

---

## 💾 Base de Dados & Domínio do Problema
| Fonte | Tipo | Atributos (1 Hz) |
|-------|------|------------------|
| **Sintético** | Solução numérica da equação de calor 1‑D | _t, I(t), T_surface(t), T_internal(t)_ |
| **Real ("Electric Motor Temperature")** | Kaggle: wkirgsn/electric-motor-temperature | _u_d, u_q, i_d, i_q, motor_speed, torque, pm, stator_yoke, stator_winding, ambient, coolant_ |

Condições de contorno: \(u(0,t)=T_{surface}(t)\); \(∂u/∂x|_{x=L}=0\).

### Integração com o Dataset Electric Motor Temperature
As medições reais para os experimentos **E2** são obtidas do dataset [Electric Motor Temperature](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature). O arquivo principal `pmsm_temperature_data.csv` contém correntes, tensões e diversas temperaturas registradas em um motor PMSM. O carregamento pode ser feito com `kagglehub`:

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

real_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "wkirgsn/electric-motor-temperature",
    "pmsm_temperature_data.csv",
)
```

Após o download, o dataset é dividido em subconjuntos de **validação** e **teste** conforme o planejamento experimental.

---

## 🛠️ Pré‑processamento & Geração de Dados Sintéticos
1. **Parâmetros aproximados:** \(R=2.3 Ω\), \(α=1.1×10^{-4} m²/s\), \(L=0.02 m\).  
2. **Perfis de corrente:** degraus, rampas, e ciclos extraídos de registros reais.  
3. **Ruído aditivo:** \(N(0,0.5 °C)\) em temperaturas; ±2 % em corrente.  
4. **Normalização:** Min‑Max \([0,1]\).  
5. **Divisão temporal:** 70 % treino, 20 % validação, 10 % teste.

---

## 🏗️ Arquitetura PINN
| Bloco | Configuração |
|-------|--------------|
| **Entrada** | \((x,t)\) |
| **Hidden** | 6 × 64 neurônios, `tanh` |
| **Saída**  | \(u_θ(x,t)\) (temperatura) |
| **Loss**   | \(𝓛 = λ_f‖PDE‖² + λ_b‖BC‖² + λ_d‖Dados‖²\) |
| **Optimizador** | Adam 1e‑3 → L‑BFGS |

---

## 🧪 Planejamento Experimental
| Experimento | Objetivo | Dados |
|-------------|----------|-------|
| **E1** | Verificar se o PINN aprende a PDE (sintético) | Sintético |
| **E2** | Ajustar parâmetros \(α,R\) via fine‑tuning | Real (validação) |
| **E3** | Inferência pura em turno inédito | Real (teste) |

Métricas: **MAE**, **RMSE**, **ρ de Pearson**.

---

## 🔬 Experimentos & Resultados
*(preencher após execução no notebook)*
Os experimentos seguem o planejamento da tabela anterior e utilizam o dataset **Electric Motor Temperature** para os passos de validação e teste:

- **E1 – Sintético:** verifica se o PINN aprende corretamente a PDE gerada artificialmente.
- **E2 – Validação:** fine‑tuning do modelo com uma fração do dataset real.
- **E3 – Teste:** avaliação final em um turno inédito.

As métricas observadas são **MAE**, **RMSE** e **coeficiente de Pearson** entre temperatura prevista e medida.

---

## 📊 Análise e Discussão
*(discutir precisão, sensibilidade a α e R, limitações e extensões)*

---

## 🔚 Considerações Finais
1. PINN estima \(T_{internal}\) com erro ≤ __ °C usando somente \(I\) e \(T_{surface}\).  
2. Metodologia viabiliza implantação **shadow‑mode** sem intervenção física.  
3. Próximos passos: incorporar vibração RMS e estender modelo para 2‑D radial.

---

## 🚀 Reprodutibilidade
Todo o código está centralizado no **notebook `pinn_motor_thermal.ipynb`**. Para executá‑lo:

```bash
git clone https://github.com/thalesfb/machine_learning/
cd pinn
python -m venv .venv
source .venv/bin/activate
jupyter pinn_motor_thermal.ipynb
```

> **Google Colab:** basta clicar em `Open in Colab` no topo do notebook.

---

## 📚 Referências
> **Formato ABNT (NBR 6023:2018)**  

[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics‑informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019, doi: https://doi.org/10.1016/j.jcp.2018.10.045.  

[2] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics‑informed deep learning (Part I): Data‑driven solutions of nonlinear partial differential equations,” *arXiv* preprint arXiv:1711.10561, 2017. Available: https://arxiv.org/abs/1711.10561  

[3] S. Cuomo *et al.*, “Scientific machine learning through physics‑informed neural networks: Where we are and what’s next,” *Journal of Scientific Computing*, vol. 92, art. 88, 2022, doi: https://doi.org/10.1007/s10915-022-01939-z.  

[4] Y. Zhang *et al.*, “Physics‑informed neuro‑evolution (PINE): A survey and prospects,” *arXiv* preprint arXiv:2501.06572, 2025. Available: https://arxiv.org/abs/2501.06572  

[5] J. L. Öberg, *Physics‑Informed Neural Network for Thermal Modeling of an Electric Motor Drive*, M.S. thesis, Chalmers Univ. of Technology, Gothenburg, Sweden, 2023. Available: https://odr.chalmers.se/items/03b63aad-812d-4ec3-9679-1aa65981eff6  

[6] T. Nguyen, J. Lee, and K. Park, “End‑to‑end differentiable physics temperature estimation for permanent‑magnet synchronous motor drives,” *Sensors*, vol. 23, no. 4, p. 174, 2023, doi: https://doi.org/10.3390/s23040174.  

[7] L. Eriksson, *Online Temperature Prediction in Electric Machines Using PINNs*, M.S. thesis, KTH Royal Institute of Technology, Stockholm, Sweden, 2024. Available: https://kth.diva-portal.org/smash/get/diva2:1749477/FULLTEXT02.pdf  

[8] L. Glass, W. Hilali, and O. Nelles, “An input‑to‑state stable virtual sensor for electric motor rotor temperature,” *IFAC‑PapersOnLine*, vol. 56, no. 1, pp. 240–245, 2023, doi: https://doi.org/10.1016/j.ifacol.2023.10.040.  

[9] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis, “DeepXDE: A deep learning library for solving differential equations,” *SIAM Review*, vol. 63, no. 1, pp. 208–228, 2021, doi: https://doi.org/10.1137/19M1274067.  

[10] E. Haghighat and R. Juanes, “SciANN: A Keras/TensorFlow wrapper for scientific computations and physics‑informed deep learning using artificial neural networks,” *Computer Methods in Applied Mechanics and Engineering*, vol. 373, art. 113552, 2021, doi: https://doi.org/10.1016/j.cma.2020.113552.  

[11] E. Torres, J. Schiefer, and M. Niepert, “Adaptive physics‑informed neural networks: A survey,” *arXiv* preprint arXiv:2503.18181, 2025. Available: https://arxiv.org/abs/2503.18181  

[12] E. Haghighat and R. Juanes, *SciANN Documentation*, 2024. [Online]. Available: https://www.sciann.com/ 

---

> _“A física ensina; a rede aprende.”_