# GOK_SBRC_2025


### Bibliotecas necessárioas: 


## Estrutura do Projeto

### Pastas e Arquivos
Dentro da pasta src vai encontrar os seguintes arquivos/pastas:
- **`anon.py`**: Responsável pelo processo de anonimização específico.
- **`{nome_do_algoritmo_anonimização}.py`**: Arquivo principal que executa o algoritmo de anonimização.
- **`ml.py`**: Responsável pelo treinamento de modelos de aprendizado de máquina.
- **`config`** : Pasta que guarda a especificação do dataset e dos quasi identificadores específicos



## Como Usar

1. **Configuração**: Edite o arquivo `config` para especificar o dataset e os quasi-identificadores, caso querira trocar os que já tem .
2. **Anonimização**: Execute `anon.py` para processar e anonimizar os dados, pois ele executurá o processo de anon imização e utilizará os arquivos da pasta config.
3. **Treinamento**: Execute `ml.py` para treinar modelos de aprendizado de máquina com os dados anonimizados com o algorimo específico.

## Exemplo de Uso

```bash
python anon.py
python ml.py
