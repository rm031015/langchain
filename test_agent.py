from agents.linkedin_lookup_angent import lookup as linkedin_lookup_angent
from third_parties.linkedin import scrape_linkedin_profile
from informationClass import InformationProcessor
from langchain.prompts.prompt import PromptTemplate
if __name__ == "__main__":

    linkedin_username = linkedin_lookup_angent(
        name='Wan-Lin Chou Senior Information Engineer ')
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_username)
    processor = InformationProcessor(
        information=linkedin_data, model_name="gpt-3.5-turbo", temperature=0)
    prompt_template = processor.create_prompt_template()
    processor.create_llm_chain(prompt_template)
    result = processor.process_information()
    print(result)
