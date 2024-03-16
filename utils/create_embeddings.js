import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { readFile } from "fs/promises";
import 'dotenv/config.js'
import { createClient } from "@supabase/supabase-js";
import { OpenAIEmbeddings } from "@langchain/openai";
import {SupabaseVectorStore} from "@langchain/community/vectorstores/supabase"

try {
  const text = await readFile('info.txt','utf-8')

  const splitter = new RecursiveCharacterTextSplitter();

  const output = await splitter.createDocuments([text]);

  const sbApiKey = process.env.SUPABASE_API_KEY
  const sbUrl = process.env.SUPABASE_URL
  const openAIApiKey = process.env.OPENAI_API_KEY

  const client = createClient(sbUrl,sbApiKey)

  await SupabaseVectorStore.fromDocuments(
    output,
    new OpenAIEmbeddings({openAIApiKey}),
    {
      client,
      tableName: 'documents',
    }
  )

} catch (e) {
  console.error(e);
}
