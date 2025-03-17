// import 'dotenv/config';
// import express from 'express';
// import cors from 'cors';
// import { StateGraph } from "@langchain/langgraph";
// import { GoogleGenerativeAI } from '@google/generative-ai';
// import { ChatPromptTemplate } from "@langchain/core/prompts";
// import { RunnableSequence } from "@langchain/core/runnables";
// import { Pinecone } from '@pinecone-database/pinecone';

// const app = express();
// app.use(cors());
// app.use(express.json());

// // // Initialize services
// const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
// const index = pinecone.Index("careerpeek");

// const graph = new StateGraph({
//   channels: {
//     missingFields: {
//       value: () => [],
//       aggregate: (a, b) => [...a, ...b], // Array concatenation
//     },
//     studentData: {
//       value: () => ({}),
//       aggregate: (a, b) => ({ ...a, ...b }), // Object merge
//     },
//     question: {
//       value: () => null,
//       aggregate: (a, b) => b, // Last write wins
//     },
//     userResponse: {
//       value: () => null,
//       aggregate: (a, b) => b, // Last write wins
//     }
//   }
// });

// // Initialize services
// const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);

// // ✅ PROPERLY DEFINE BOTH MODELS
// const textModel = genAI.getGenerativeModel({ 
//   model: "gemini-pro" 
// });

// const embeddingModel = genAI.getGenerativeModel({
//   model: "text-embedding-004"
// });

// // Rest of your initialization code (Pinecone, Express, etc.)

// // ✅ CORRECTED store_in_pinecone NODE
// graph.addNode("store_in_pinecone", async (state) => {
//   const textData = JSON.stringify(state.studentData);
  
//   // Use the embedding model
//   const embeddingResult = await embeddingModel.embedContent(textData);
//   const embedding = embeddingResult.embedding.values;

//   await index.upsert([{
//     id: `student_${Date.now()}`,
//     values: embedding,
//     metadata: state.studentData
//   }]);

//   return state;
// });
// // Define state structure
// const requiredFields = ["name", "age", "academic_history", "aspirations", "skills", "dreams"];

// // ✅ Correct State Graph Initialization
// // Replace the existing graph initialization with:

// // ✅ Define Nodes Correctly
// graph.addNode("start", async (state) => {
//     return { missingFields: [...requiredFields], studentData: {} };
// });

// graph.addNode("check_missing_info", async (state) => {
//     const missing = requiredFields.filter(f => !state.studentData[f]);
//     return { missingFields: missing };
// });

// graph.addNode("ask_question", async (state) => {
//     const nextField = state.missingFields[0];
//     return { question: `Please share your ${nextField}:` };
// });

// graph.addNode("store_answer", async (state) => {
//     const currentField = state.missingFields[0];
//     return {
//       studentData: {
//         ...state.studentData,
//         [currentField]: state.userResponse
//       },
//       missingFields: state.missingFields.slice(1),
//       userResponse: null
//     };
//   });


// graph.addNode("career_decision", async (state) => {
//   const prompt = ChatPromptTemplate.fromMessages([
//     ["system", "You are a career advisor specializing in Indian education systems. Suggest career paths based on:"],
//     ["human", `### Student Profile:
//       Name: {name}
//       Age: {age}
//       Education: {academic_history}
//       Skills: {skills}
//       Goals: {aspirations}
//       Dreams: {dreams}`]
//     ]);
    
//     const chain = RunnableSequence.from([
//         prompt,
//         async (input) => (await textModel.generateContent(input)).response.text()
//     ]);
// // Then invoke with:
// chain.invoke({
//     name: state.studentData.name,
//     age: state.studentData.age,
//     academic_history: state.studentData.academic_history,
//     skills: state.studentData.skills,
//     aspirations: state.studentData.aspirations,
//     dreams: state.studentData.dreams
// });
  
  
//   // Pass variables explicitly
//   return { 
//       recommendation: await chain.invoke({
//           studentData: JSON.stringify(state.studentData)
//       }) 
//   };
// });

// // ✅ Define Edges Correctly
// graph.setEntryPoint("start");
// graph.addEdge("start", "check_missing_info");
// graph.addConditionalEdges(
//     "check_missing_info",
//     (state) => state.missingFields.length > 0 ? "ask_question" : "store_in_pinecone"
// );
// graph.addEdge("ask_question", "store_answer");
// graph.addEdge("store_answer", "check_missing_info");
// graph.addEdge("store_in_pinecone", "career_decision");

// // API Endpoint
// const conversations = new Map();

// app.post('/chat', async (req, res) => {
//     const { conversationId, message } = req.body;
    
//     try {
//       const app = await graph.compile();
      
//       if (!conversationId) {
//         const newId = Date.now().toString();
//         const initialStep = await app.invoke({
//           missingFields: requiredFields,
//           studentData: {}
//         });
//         conversations.set(newId, initialStep);
//         return res.json({
//           conversationId: newId,
//           question: initialStep.question
//         });
//       }
  
//       const prevState = conversations.get(conversationId) || {
//         missingFields: requiredFields,
//         studentData: {}
//       };
  
//       const nextStep = await app.invoke({
//         ...prevState,
//         userResponse: message
//       });
  
//       conversations.set(conversationId, nextStep);
  
//       res.json({
//         question: nextStep.question,
//         recommendation: nextStep.recommendation,
//         completed: !nextStep.question
//       });
//     } catch (error) {
//       console.error("Error:", error);
//       res.status(500).json({ 
//         error: error.message,
//         stack: error.stack 
//       });
//     }
//   });

// app.listen(3000, () => console.log('Server running on port 3000'));
// import 'dotenv/config';
// import express from 'express';
// import cors from 'cors';
// import { StateGraph } from "@langchain/langgraph";
// import { GoogleGenerativeAI } from '@google/generative-ai';
// import { ChatPromptTemplate } from "@langchain/core/prompts";
// import { RunnableSequence } from "@langchain/core/runnables";
// import { Pinecone } from '@pinecone-database/pinecone';
// import { VertexAI } from '@google-cloud/vertexai';
// import { pipeline } from "@xenova/transformers";

// const app = express();
// app.use(cors());
// app.use(express.json());

// // Ensure environment variables are set
// if (!process.env.GOOGLE_CLOUD_PROJECT_ID) {
//   console.error("Missing GOOGLE_CLOUD_PROJECT_ID in environment variables.");
//   process.exit(1);
// }
// if (!process.env.PINECONE_API_KEY) {
//   console.error("Missing PINECONE_API_KEY in environment variables.");
//   process.exit(1);
// }

// // Initialize Pinecone
// const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
// const index = pinecone.Index("careerpeek");

// // Initialize Vertex AI properly
// const vertexAI = new VertexAI({
//   project: process.env.GOOGLE_CLOUD_PROJECT_ID,
//   location: 'us-central1'
// });

// const model = vertexAI.getGenerativeModel({
//   model: 'gemini-pro',
//   generation_config: {
//       max_output_tokens: 2048,
//       temperature: 0.9,
//       top_p: 1,
//   }
// });

// // Define Embedding Function
// async function getEmbedding(text) {
//   const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
//   const result = await embedder(text, { pooling: "mean", normalize: true, dtype: 'float32' });
//   return Array.from(result.data);
// }

// // Initialize StateGraph
// const graph = new StateGraph({
//   channels: {
//     missingFields: { value: () => [], aggregate: (a, b) => [...a, ...b] },
//     studentData: { value: () => ({}), aggregate: (a, b) => ({ ...a, ...b }) },
//     question: { value: () => null, aggregate: (a, b) => b },
//     userResponse: { value: () => null, aggregate: (a, b) => b },
//     recommendation: { value: () => null, aggregate: (a, b) => b }
//   }
// });

// // Define Nodes
// const requiredFields = ["name", "age", "academic_history", "aspirations", "skills", "dreams"];

// graph.addNode("start", async () => ({ missingFields: [...requiredFields], studentData: {} }));

// graph.addNode("check_missing_info", async (state) => ({
//   missingFields: requiredFields.filter(f => !state.studentData[f])
// }));

// graph.addNode("ask_question", async (state) => {
//   if (!state.missingFields.length) {
//       return { question: null }; // No more questions
//   }
//   const nextField = state.missingFields[0];
//   return { question: `Please share your ${nextField}:` };
// });

// graph.addNode("store_answer", async (state) => ({
//   studentData: { ...state.studentData, [state.missingFields[0]]: state.userResponse },
//   missingFields: state.missingFields.slice(1),
//   userResponse: null
// }));

// graph.addNode("store_in_pinecone", async (state) => {
//   try {
//     const textData = JSON.stringify(state.studentData);
//     const embedding = await getEmbedding(textData);
    
//     if (!Array.isArray(embedding) || embedding.some(isNaN)) {
//       throw new Error("Invalid embedding format");
//     }

//     await index.upsert([{ id: `student_${Date.now()}`, values: embedding, metadata: state.studentData }]);

//     return state;
//   } catch (error) {
//     console.error("Embedding Error:", error);
//     throw error;
//   }
// });

// app.post("/chat", async (req, res) => {
//   try {
//       const { name, message } = req.body;
//       const formattedInput = {
//           contents: [{ parts: [{ text: message }], role: "user" }]
//       };

//       const result = await model.generateContent(formattedInput);

//       console.log("Full Response:", JSON.stringify(result, null, 2));

//       const responseText = result.candidates?.[0]?.content?.parts?.[0]?.text || "No response generated.";

//       res.json({
//           conversationId: "1742231879456",
//           response: responseText
//       });
//   } catch (error) {
//       console.error("Error in chat:", error);
//       res.status(500).json({ error: "Internal Server Error" });
//   }
// });


//   const chain = RunnableSequence.from([
//     prompt,
//     async (input) => {
//       const messages = await prompt.formatMessages(studentData);
//       const formattedInput = {
//         contents: messages.map(msg => ({
//           parts: [{ text: msg.content }],
//           role: msg._getType() === "SystemMessage" ? "system" : "user"
//         }))
//       };

//       try {
//         const result = await model.generateContent(formattedInput);
        
//         // Log full response
//         console.log("Full Response:", JSON.stringify(result, null, 2));

//         // Extract and log candidates
//         if (result && result.candidates) {
//           console.log("Candidates:", JSON.stringify(result.candidates, null, 2));
//         }

//         return result.response?.candidates[0]?.content?.parts[0]?.text || "No response";
//       } catch (error) {
//         console.error("Vertex AI Error:", error);
//         return "Error generating career advice.";
//       }
//     }
//   ]);

//   return { recommendation: await chain.invoke(studentData) };
// });

// // Define Graph Flow
// graph.setEntryPoint("start");
// graph.addEdge("start", "check_missing_info");
// graph.addConditionalEdges("check_missing_info", (state) => state.missingFields.length > 0 ? "ask_question" : "store_in_pinecone");
// graph.addEdge("ask_question", "store_answer");
// graph.addEdge("store_answer", "check_missing_info");
// graph.addEdge("store_in_pinecone", "career_decision");

// // API Handling
// const conversations = new Map();

// app.post('/chat', async (req, res) => {
//   const { conversationId, message } = req.body;
//   try {
//       const app = await graph.compile();

//       if (!conversationId) {
//           const newId = Date.now().toString();
//           const initialState = { 
//               missingFields: [...requiredFields], 
//               studentData: {} 
//           };
//           const initialStep = await app.invoke(initialState);
//           conversations.set(newId, initialStep);

//           return res.json({ 
//               conversationId: newId, 
//               question: initialStep.question 
//           });
//       }

//       const prevState = conversations.get(conversationId) || { 
//           missingFields: [...requiredFields], 
//           studentData: {} 
//       };
//       const nextStep = await app.invoke({ 
//           ...prevState, 
//           userResponse: message 
//       });

//       conversations.set(conversationId, nextStep);
//       res.json({
//           conversationId,
//           question: nextStep.question || "Processing complete",
//           recommendation: nextStep.recommendation || null,
//           completed: nextStep.missingFields.length === 0
//       });

//   } catch (error) {
//       console.error("Error:", error);
//       res.status(500).json({ error: error.message, stack: error.stack });
//   }
// });


// // Corrected `makeRequest()` function
// async function makeRequest() {
//   try {
//     const prompt = { contents: [{ parts: [{ text: "Suggest a career for a student with a background in AI and ML." }], role: "user" }] };
//     const response = await model.generateContent(prompt);
//     console.log("Response:", response);
//   } catch (error) {
//     console.error("Request Error:", error);
//   }
// }

// makeRequest();

// // Start Server
// app.listen(3000, () => console.log('Server running on port 3000'));






// import 'dotenv/config';
// import express from 'express';
// import cors from 'cors';
// import { StateGraph } from "@langchain/langgraph";
// import { GoogleGenerativeAI } from '@google/generative-ai';
// import { Pinecone } from '@pinecone-database/pinecone';
// import { VertexAI } from '@google-cloud/vertexai';
// import { pipeline } from "@xenova/transformers";

// const app = express();
// app.use(cors());
// app.use(express.json());

// if (!process.env.GOOGLE_CLOUD_PROJECT_ID || !process.env.PINECONE_API_KEY) {
//   console.error("Missing required environment variables.");
//   process.exit(1);
// }

// const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
// const index = pinecone.Index("careerpeek");

// const vertexAI = new VertexAI({
//   project: process.env.GOOGLE_CLOUD_PROJECT_ID,
//   location: 'us-central1'
// });

// const model = vertexAI.getGenerativeModel({
//   model: 'gemini-pro',
//   generation_config: {
//       max_output_tokens: 2048,
//       temperature: 0.9,
//       top_p: 1,
//   }
// });

// async function getEmbedding(text) {
//   const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
//   const result = await embedder(text, { pooling: "mean", normalize: true, dtype: 'float32' });
//   return Array.from(result.data);
// }

// const graph = new StateGraph({
//   channels: {
//     missingFields: { value: () => [], aggregate: (a, b) => [...a, ...b] },
//     studentData: { value: () => ({}), aggregate: (a, b) => ({ ...a, ...b }) },
//     question: { value: () => null, aggregate: (a, b) => b },
//     userResponse: { value: () => null, aggregate: (a, b) => b },
//     recommendation: { value: () => null, aggregate: (a, b) => b }
//   }
// });

// const requiredFields = ["name", "age", "academic_history", "aspirations", "skills", "dreams"];

// graph.addNode("start", async () => ({ missingFields: [...requiredFields], studentData: {} }));

// graph.addNode("check_missing_info", async (state) => ({
//   missingFields: requiredFields.filter(f => !state.studentData[f])
// }));

// graph.addNode("ask_question", async (state) => {
//   if (!state.missingFields.length) return { question: null };
//   const nextField = state.missingFields[0];
//   return { question: `Please share your ${nextField}:` };
// });

// graph.addNode("store_answer", async (state) => ({
//   studentData: { ...state.studentData, [state.missingFields[0]]: state.userResponse },
//   missingFields: state.missingFields.slice(1),
//   userResponse: null
// }));

// graph.addNode("store_in_pinecone", async (state) => {
//   try {
//     const textData = JSON.stringify(state.studentData);
//     const embedding = await getEmbedding(textData);
//     if (!Array.isArray(embedding) || embedding.some(isNaN)) throw new Error("Invalid embedding format");
//     await index.upsert([{ id: `student_${Date.now()}`, values: embedding, metadata: state.studentData }]);
//     return state;
//   } catch (error) {
//     console.error("Embedding Error:", error);
//     throw error;
//   }
// });
// graph.addNode("career_decision", async (state) => {
//   return { recommendation: "Career decision processing completed!" };
// });

// graph.setEntryPoint("start");
// graph.addEdge("start", "check_missing_info");
// graph.addConditionalEdges("check_missing_info", (state) => state.missingFields.length > 0 ? "ask_question" : "store_in_pinecone");
// graph.addEdge("ask_question", "store_answer");
// graph.addEdge("store_answer", "check_missing_info");
// graph.addEdge("store_in_pinecone", "career_decision");

// const conversations = new Map();

// app.post("/chat", async (req, res) => {
//   try {
//       const { name, message } = req.body;
//       const formattedInput = {
//           contents: [{ parts: [{ text: message }], role: "user" }]
//       };

//       const result = await model.generateContent(formattedInput);

//       // Debugging: Log the full response
//       console.log("Full Response:", JSON.stringify(result, null, 2));

//       // Extract the first candidate's text response
//       let responseText = "No response generated.";
//       if (result.candidates && result.candidates.length > 0) {
//           responseText = result.candidates[0]?.content?.parts
//               .map(part => part.text) // Extract all text parts
//               .join(" "); // Combine them into a single string
//       }

//       res.json({
//           conversationId: "1742231879456",
//           response: responseText
//       });
//   } catch (error) {
//       console.error("Error in chat:", error);
//       res.status(500).json({ error: "Internal Server Error" });
//   }
// });

// async function makeRequest() {
//   try {
//     const prompt = { contents: [{ parts: [{ text: "Suggest a career for a student with a background in AI and ML." }], role: "user" }] };
//     const response = await model.generateContent(prompt);
//     console.log("Response:", response);
//   } catch (error) {
//     console.error("Request Error:", error);
//   }
// }

// makeRequest();

// app.listen(3000, () => console.log('Server running on port 3000'));





import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { StateGraph } from "@langchain/langgraph";
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Pinecone } from '@pinecone-database/pinecone';
import { VertexAI } from '@google-cloud/vertexai';
import { pipeline } from "@xenova/transformers";

const app = express();
app.use(cors());
app.use(express.json());

if (!process.env.GOOGLE_CLOUD_PROJECT_ID || !process.env.PINECONE_API_KEY) {
  console.error("Missing required environment variables.");
  process.exit(1);
}

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.Index("careerpeek");

const vertexAI = new VertexAI({
  project: process.env.GOOGLE_CLOUD_PROJECT_ID,
  location: 'us-central1'
});

const model = vertexAI.getGenerativeModel({
  model: 'gemini-pro',
  generation_config: {
      max_output_tokens: 2048,
      temperature: 0.9,
      top_p: 1,
  }
});

async function getEmbedding(text) {
  const embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  const result = await embedder(text, { pooling: "mean", normalize: true, dtype: 'float32' });
  return Array.from(result.data);
}

// Define state graph
const graph = new StateGraph({
  channels: {
    missingFields: { value: () => [], aggregate: (a, b) => [...a, ...b] },
    studentData: { value: () => ({}), aggregate: (a, b) => ({ ...a, ...b }) },
    question: { value: () => null, aggregate: (a, b) => b },
    userResponse: { value: () => null, aggregate: (a, b) => b },
    recommendation: { value: () => null, aggregate: (a, b) => b }
  }
});
const compiledGraph = graph.compile(); // ✅ Compile the graph

const requiredFields = ["name", "age", "academic_history", "aspirations", "skills", "dreams"];

graph.addNode("start", async () => ({ missingFields: [...requiredFields], studentData: {} }));

graph.addNode("check_missing_info", async (state) => ({
  missingFields: requiredFields.filter(f => !state.studentData[f])
}));

graph.addNode("ask_question", async (state) => {
  if (!state.missingFields.length) return { question: null };
  const nextField = state.missingFields[0];
  return { question: `Please share your ${nextField}:` };
});

graph.addNode("store_answer", async (state) => ({
  studentData: { ...state.studentData, [state.missingFields[0]]: state.userResponse },
  missingFields: state.missingFields.slice(1),
  userResponse: null
}));

graph.addNode("store_in_pinecone", async (state) => {
  try {
    const textData = JSON.stringify(state.studentData);
    const embedding = await getEmbedding(textData);
    if (!Array.isArray(embedding) || embedding.some(isNaN)) throw new Error("Invalid embedding format");
    
    await index.upsert([{ id: `student_${Date.now()}`, values: embedding, metadata: state.studentData }]);
    return state;
  } catch (error) {
    console.error("Embedding Error:", error);
    throw error;
  }
});

graph.addNode("career_decision", async (state) => {
  return { recommendation: `Based on your profile, a suitable career path could be ${state.studentData.aspirations} with a focus on ${state.studentData.skills}.` };
});

graph.setEntryPoint("start");
graph.addEdge("start", "check_missing_info");
graph.addConditionalEdges("check_missing_info", (state) => state.missingFields.length > 0 ? "ask_question" : "store_in_pinecone");
graph.addEdge("ask_question", "store_answer");
graph.addEdge("store_answer", "check_missing_info");
graph.addEdge("store_in_pinecone", "career_decision");

// Store conversations
const conversations = new Map();

app.post("/chat", async (req, res) => {
  try {
    const { conversationId, userInput } = req.body;

    let state = conversations.get(conversationId) || { userResponse: userInput };

    state = await compiledGraph.invoke(state); // ✅ Use compiled graph

    let responseText = "";
    
    if (state.question) {
      responseText = state.question;
    } else if (state.recommendation) {
      responseText = state.recommendation;
    } else {
      responseText = "Processing your information...";
    }

    // ✅ Convert object to text if necessary
    if (typeof responseText === "object") {
      responseText = JSON.stringify(responseText, null, 2);
    }

    res.json({ conversationId, response: responseText });

    conversations.set(conversationId, state);
  } catch (error) {
    console.error("Error in chat:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});


// Test AI response
async function makeRequest() {
  try {
    const prompt = { contents: [{ parts: [{ text: "Suggest a career for a student with a background in AI and ML." }], role: "user" }] };
    const response = await model.generateContent(prompt);

    // ✅ Ensure response is converted to text
    let responseText = response?.candidates?.[0]?.content || "No response received.";

    if (typeof responseText === "object") {
      responseText = JSON.stringify(responseText, null, 2);
    }

    console.log("Response:", responseText);
  } catch (error) {
    console.error("Request Error:", error);
  }
}

makeRequest();

app.listen(3000, () => console.log('Server running on port 3000'));
