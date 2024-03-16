import "dotenv/config.js";
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { retriever } from "./utils/retriever.js";
import { combineDocuments } from "./utils/combine_documents.js";
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import * as readline from 'readline-sync'


const openAIApiKey = process.env.OPENAI_API_KEY;

const llm = new ChatOpenAI({ openAIApiKey });

const standaloneQuestionTemplate = `given a question, convert it into a standalone question.
question: {question} standalone question:`;

const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
  standaloneQuestionTemplate
);

const standaloneQuestionChain = RunnableSequence.from([
  standaloneQuestionPrompt,
  llm,
  new StringOutputParser()
]);

const retrieverChain = RunnableSequence.from([
  prevResult => prevResult.standalone_question,
  retriever,
  combineDocuments,
  new StringOutputParser()
]);

const answerTemplate = `Answer question based on the provided context. If you
cannot find the answer in the context, apologize and tell the user to email
dsc@ucsd.edu. Speak in a friendly tone.

context: {context}
question: {question}
answer: `;

const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);

const answerChain = RunnableSequence.from([
    answerPrompt,
    llm,
    new StringOutputParser()
]);

const chain = RunnableSequence.from([
  { standalone_question: standaloneQuestionChain,
    original_input: new RunnablePassthrough()
    },
  {
    context: retrieverChain,
    question: ({original_input}) => original_input.question,
  },
  answerChain
]);

const prompt = readline.question('What is your question? ')

const response = await chain.invoke({
    question: prompt
  });

console.log(response)