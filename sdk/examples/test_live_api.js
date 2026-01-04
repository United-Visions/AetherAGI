/**
 * Live API Test - JavaScript SDK
 * Tests real connection to AetherMind API
 */

const { AetherMindClient, AetherMindError } = require('../javascript/dist/index.js');

// Your live API key
const API_KEY = 'am_live_QMEoiMVz2jdZJ_EBJ951YuzBseCrhsgXu2mHITFdQZ4';

async function testLiveAPI() {
  console.log('='.repeat(60));
  console.log('🚀 Testing AetherMind JavaScript SDK with Live API');
  console.log('='.repeat(60));
  console.log();

  // Initialize client
  console.log('📡 Initializing client with live API key...');
  const client = new AetherMindClient({
    apiKey: API_KEY,
    baseURL: 'https://aethermind-frontend.onrender.com'  // Your API endpoint
  });
  console.log('✅ Client initialized!');
  console.log();

  // Test 1: Simple chat
  console.log('='.repeat(60));
  console.log('Test 1: Hello AetherMind Chat');
  console.log('='.repeat(60));
  console.log();

  try {
    console.log('💬 Sending: "Hello AetherMind! Can you hear me?"');
    const response = await client.chat({
      message: 'Hello AetherMind! Can you hear me?'
    });

    console.log('\n🤖 Response:');
    console.log(`   Answer: ${response.answer || 'No answer'}`);

    if (response.confidence) {
      console.log(`   Confidence: ${(response.confidence * 100).toFixed(0)}%`);
    }

    if (response.tokens_used) {
      console.log(`   Tokens: ${response.tokens_used}`);
    }

    console.log('\n✅ Chat API works!');

  } catch (error) {
    console.error(`\n❌ Error: ${error.message}`);
    console.log('\nNote: Make sure your API endpoint is running at:');
    console.log('https://aethermind-frontend.onrender.com');
  }

  console.log();
  console.log('='.repeat(60));
  console.log('🎉 Live API Test Complete!');
  console.log('='.repeat(60));
}

// Run test
testLiveAPI().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
