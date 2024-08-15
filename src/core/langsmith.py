import os
from langchain import evaluation
from langchain.callbacks import LangChainTracer
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

class LangSmithIntegration:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("LANGCHAIN_API_KEY")
        self.project = os.getenv("LANGCHAIN_PROJECT", "ai-nirvana")
        self.tracer = LangChainTracer(project_name=self.project)
        self.llm = ChatOpenAI(
            model_name=self.config.MODEL_NAME,
            openai_api_key=self.config.API_KEY,
            openai_api_base=self.config.API_BASE,
            temperature=self.config.TEMPERATURE,
            callbacks=[self.tracer]
        )

    def create_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant named AI Nirvana. Be helpful, concise, and clear."),
            ("human", "{input}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain

    def run_chain(self, input_text):
        chain = self.create_chain()
        return chain.invoke({"input": input_text})

    def create_dataset(self, examples):
        client = Client()
        dataset_name = f"{self.project}-dataset"
        dataset = client.create_dataset(dataset_name)
        for example in examples:
            client.create_example(inputs=example["input"], outputs=example["output"], dataset_id=dataset.id)
        return dataset

    def evaluate_chain(self, dataset_name):
        eval_config = RunEvalConfig(
            evaluators=[
                evaluation.load_evaluator("criteria"),
                evaluation.load_evaluator("qa"),
                evaluation.load_evaluator("context_qa"),
            ]
        )
        chain = self.create_chain()
        run_on_dataset(
            client=LangChainTracer(project_name=self.project),
            dataset_name=dataset_name,
            llm_or_chain_factory=lambda: chain,
            evaluation=eval_config,
        )

    def setup_retrieval_qa(self, documents):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(texts, embeddings)
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=docsearch.as_retriever())
        return qa

    def answer_question(self, qa, question):
        return qa.run(question)

    def get_evaluation_results(self, run_id):
        client = Client()
        run = client.read_run(run_id)
        return run.feedback

    def analyze_chain_performance(self, dataset_name):
        client = Client()
        dataset = client.read_dataset(dataset_name=dataset_name)
        runs = client.list_runs(dataset_id=dataset.id, execution_order=1)
        
        total_runs = 0
        successful_runs = 0
        average_latency = 0
        
        for run in runs:
            total_runs += 1
            if run.error is None:
                successful_runs += 1
                average_latency += run.latency
        
        if successful_runs > 0:
            average_latency /= successful_runs
        
        success_rate = (successful_runs / total_runs) * 100 if total_runs > 0 else 0
        
        return {
            "total_runs": total_runs,
            "success_rate": success_rate,
            "average_latency": average_latency
        }

    def optimize_prompt(self, base_prompt, dataset_name, num_iterations=5):
        best_prompt = base_prompt
        best_performance = 0
        client = Client()  # 使用 langsmith 的 Client
    
        for _ in range(num_iterations):
            current_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant named AI Nirvana. Be helpful, concise, and clear."),
                ("human", best_prompt)
            ])
            chain = current_prompt | self.llm | StrOutputParser()
        
            run_on_dataset(
                client=client,  # 使用 langsmith 的 Client
                dataset_name=dataset_name,
                llm_or_chain_factory=lambda: chain,
                evaluation=RunEvalConfig(evaluators=[evaluation.load_evaluator("criteria")])
            )
        
            performance = self.analyze_chain_performance(dataset_name)
        
            if performance["success_rate"] > best_performance:
                best_prompt = current_prompt
                best_performance = performance["success_rate"]
        
            improvement_prompt = f"Improve this prompt for better performance: {best_prompt}"
            best_prompt = self.run_chain(improvement_prompt)
    
        return best_prompt

    def continuous_learning(self, input_text, output_text, feedback):
        client = Client()
        
        run = client.create_run(
            name="continuous_learning",
            inputs={"input": input_text},
            outputs={"output": output_text},
            feedback={"user_feedback": feedback}
        )
        
        if feedback == "positive":
            self.optimize_prompt(input_text, self.project)
        elif feedback == "negative":
            improvement_prompt = f"The following response was not satisfactory. How can we improve it?\nInput: {input_text}\nOutput: {output_text}"
            improved_response = self.run_chain(improvement_prompt)
            self.optimize_prompt(improved_response, self.project)

    def generate_test_cases(self, input_text):
        prompt = f"Generate 5 diverse test cases for the following input:\n{input_text}\nEach test case should include an input and an expected output."
        test_cases_text = self.run_chain(prompt)
        
        test_cases = []
        for line in test_cases_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                if key.strip().lower() == 'input':
                    test_case = {'input': value.strip()}
                elif key.strip().lower() == 'output':
                    test_case['output'] = value.strip()
                    test_cases.append(test_case)
        
        return test_cases

    def run_security_check(self, input_text):
        prompt = f"Analyze the following input for potential security risks:\n{input_text}\nProvide a security assessment and any recommendations."
        security_assessment = self.run_chain(prompt)
        return security_assessment

    def generate_explanation(self, input_text, output_text):
        prompt = f"Explain the reasoning behind this response:\nInput: {input_text}\nOutput: {output_text}\nProvide a clear and concise explanation of how the output was generated."
        explanation = self.run_chain(prompt)
        return explanation

    def integrate_with_ai_nirvana(self, ai_nirvana_instance):
        original_process = ai_nirvana_instance.process
        
        def enhanced_process(input_text):
            security_assessment = self.run_security_check(input_text)
            if "high risk" in security_assessment.lower():
                return "I'm sorry, but I cannot process this request due to security concerns."
            
            response = original_process(input_text)
            
            explanation = self.generate_explanation(input_text, response)
            
            self.continuous_learning(input_text, response, "positive")
            
            test_cases = self.generate_test_cases(input_text)
            self.create_dataset(test_cases)
            
            return f"{response}\n\nExplanation: {explanation}"
        
        ai_nirvana_instance.process = enhanced_process
        
        ai_nirvana_instance.evaluate_performance = self.analyze_chain_performance
        ai_nirvana_instance.optimize_prompts = self.optimize_prompt
        ai_nirvana_instance.setup_qa = self.setup_retrieval_qa
        ai_nirvana_instance.answer_qa = self.answer_question