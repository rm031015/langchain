from typing import Any
from informationClass import InformationProcessor
from langchain.prompts.prompt import PromptTemplate

if __name__ == "__main__":

    information = """
        Elon Reeve Musk (/ˈiːlɒn/; EE-lon; born June 28, 1971) is a businessman and investor. He is the founder, chairman, CEO, and CTO of SpaceX; angel investor, CEO, product architect and former chairman of Tesla, Inc.; owner, chairman and CTO of X Corp.; founder of the Boring Company and xAI; co-founder of Neuralink and OpenAI; and president of the Musk Foundation. He is the wealthiest person in the world, with an estimated net worth of US$232 billion as of December 2023, according to the Bloomberg Billionaires Index, and $254 billion according to Forbes, primarily from his ownership stakes in Tesla and SpaceX.[5][6]

A member of the wealthy South African Musk family, Elon was born in Pretoria and briefly attended the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada. Musk later transferred to the University of Pennsylvania, and received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University. However, Musk dropped out after two days and, with his brother Kimbal, co-founded online city guide software company Zip2. The startup was acquired by Compaq for $307 million in 1999, and, that same year Musk co-founded X.com, a direct bank. X.com merged with Confinity in 2000 to form PayPal.

In October 2002, eBay acquired PayPal for $1.5 billion, and that same year, with $100 million of the money he made, Musk founded SpaceX, a spaceflight services company. In 2004, he became an early investor in electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.). He became its chairman and product architect, assuming the position of CEO in 2008. In 2006, Musk helped create SolarCity, a solar-energy company that was acquired by Tesla in 2016 and became Tesla Energy. In 2013, he proposed a hyperloop high-speed vactrain transportation system. In 2015, he co-founded OpenAI, a nonprofit artificial intelligence research company. The following year, Musk co-founded Neuralink—a neurotechnology company developing brain–computer interfaces—and the Boring Company, a tunnel construction company. In 2022, he acquired Twitter for $44 billion. He subsequently merged the company into newly created X Corp. and rebranded the service as X the following year. In March 2023, he founded xAI, an artificial intelligence company.

    """

    # Object
    processor: InformationProcessor = InformationProcessor(
        information=information, model_name="gpt-3.5-turbo", temperature=0)
    # 1. create prompt Template
    prompt_template: PromptTemplate = processor.create_prompt_template()
    # 2. use LLM
    processor.create_llm_chain(prompt_template)
    # 3. Chain
    result: Any = processor.process_information()
    print(result)
   # result =  {'text': '1. Elon Musk is a South African-born businessman and investor known for his roles in founding and leading companies such as SpaceX, Tesla, and the Boring Company. He is also involved in various other ventures in the fields of artificial intelligence and neurotechnology.\n\n2. Two interesting facts about Elon Musk:\n- He dropped out of Stanford University after just two days to pursue his entrepreneurial ventures.\n- Musk acquired Twitter for $44 billion in 2022 and merged it into his newly created company, X Corp, rebranding the service as X the following year.'}
