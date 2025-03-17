import { Pinecone } from '@pinecone-database/pinecone';
import 'dotenv/config';
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY 
});
const indexName = 'quickstart';
const index=pc.index(indexName);
// await pc.createIndex({
//   name: indexName,
//   dimension: 3, // Replace with your model dimensions
//   metric: 'cosine', // Replace with your model metric
//   spec: { 
//     serverless: { 
//       cloud: 'aws', 
//       region: 'us-east-1' 
//     }
//   } 
// });

await index.namespace('ns1').upsert([
    {
       id: 'vec1', 
       values: [1.0, 1.5],
       metadata: { genre: 'drama' }
    },
    {
       id: 'vec2', 
       values: [2.0, 1.0],
       metadata: { genre: 'action' }
    },
    {
       id: 'vec3', 
       values: [0.1, 0.3],
       metadata: { genre: 'drama' }
    },
    {
       id: 'vec4', 
       values: [1.0, -2.5],
       metadata: { genre: 'action' }
    }
  ]);
  const response = await index.namespace('ns1').query({
    topK: 2,
    vector: [0.1, 0.3],
    includeValues: true,
    includeMetadata: true,
    filter: { genre: { '$eq': 'action' }}
  });
  
  console.log(response);