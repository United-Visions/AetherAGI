/**
 * Hello AetherMind! 🚀
 * Simple example demonstrating AetherMind AGI integration with JavaScript
 */

const { AetherMindClient } = require('@aethermind/sdk');

async function main() {
  console.log('='.repeat(60));
  console.log('🧠 Hello AetherMind - JavaScript SDK Demo');
  console.log('='.repeat(60));
  console.log();

  // Get API key from environment or use demo
  const apiKey = process.env.AETHERMIND_API_KEY || 'am_live_demo_key';

  if (apiKey === 'am_live_demo_key') {
    console.log('⚠️  Using demo API key. Set AETHERMIND_API_KEY environment variable');
    console.log('   to use your own key from https://aethermind.ai');
    console.log();
  }

  // Initialize client
  console.log('📡 Connecting to AetherMind AGI...');
  let client;
  try {
    client = new AetherMindClient({ apiKey });
    console.log('✅ Connected!\n');
  } catch (error) {
    console.error(`❌ Connection failed: ${error.message}`);
    return;
  }

  // Example 1: Basic chat
  console.log('='.repeat(60));
  console.log('Example 1: Basic Chat');
  console.log('='.repeat(60));

  const question = 'What is the difference between AI and AGI?';
  console.log(`💬 Question: ${question}`);
  console.log();

  try {
    const response = await client.chat({ message: question });
    console.log(`🤖 AetherMind: ${response.answer}`);

    if (response.confidence) {
      console.log(`📊 Confidence: ${(response.confidence * 100).toFixed(0)}%`);
    }

    if (response.reasoning_steps) {
      console.log('\n🧠 Reasoning Process:');
      response.reasoning_steps.forEach((step, i) => {
        console.log(`   ${i + 1}. ${step}`);
      });
    }
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
  }

  console.log();

  // Example 2: Domain specialist (Medical)
  console.log('='.repeat(60));
  console.log('Example 2: Medical Domain Specialist');
  console.log('='.repeat(60));

  const medicalQuestion = 'What are the symptoms of Type 2 diabetes?';
  console.log(`🏥 Medical Question: ${medicalQuestion}`);
  console.log();

  try {
    const response = await client.chat({
      message: medicalQuestion,
      namespace: 'medical'
    });
    console.log(`🤖 AetherMind (Medical): ${response.answer.substring(0, 300)}...`);
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
  }

  console.log();

  // Example 3: Memory search
  console.log('='.repeat(60));
  console.log('Example 3: Infinite Memory Search');
  console.log('='.repeat(60));

  const searchQuery = 'AGI capabilities';
  console.log(`🔍 Searching memory for: ${searchQuery}`);
  console.log();

  try {
    const memories = await client.searchMemory({
      query: searchQuery,
      topK: 3,
      includeEpisodic: true
    });

    if (memories && memories.length > 0) {
      console.log(`📚 Found ${memories.length} relevant memories:`);
      memories.forEach((memory, i) => {
        console.log(`\n   ${i + 1}. ${memory.text.substring(0, 200)}...`);
        console.log(`      Score: ${memory.score.toFixed(2)} | Time: ${memory.timestamp}`);
      });
    } else {
      console.log('   No memories found (try chatting first!)');
    }
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
  }

  console.log();

  // Example 4: Available namespaces
  console.log('='.repeat(60));
  console.log('Example 4: Available Knowledge Domains');
  console.log('='.repeat(60));

  try {
    const namespaces = await client.listNamespaces();
    console.log('🎯 Available domain specialists:');
    namespaces.forEach(ns => {
      console.log(`   - ${ns}`);
    });
  } catch (error) {
    console.log(`⚠️  Namespaces unavailable: ${error.message}`);
  }

  console.log();

  // Example 5: Usage statistics
  console.log('='.repeat(60));
  console.log('Example 5: Usage & Rate Limits');
  console.log('='.repeat(60));

  try {
    const usage = await client.getUsage();
    console.log('📊 API Usage Statistics:');
    console.log(`   Plan: ${usage.plan || 'N/A'}`);
    console.log(`   Requests Remaining: ${usage.requests_remaining || 'N/A'}`);
    console.log(`   Total Tokens Used: ${usage.total_tokens || 'N/A'}`);
    console.log(`   Reset At: ${usage.reset_at || 'N/A'}`);
  } catch (error) {
    console.log(`⚠️  Usage stats unavailable: ${error.message}`);
  }

  console.log();
  console.log('='.repeat(60));
  console.log('✨ Demo Complete!');
  console.log('='.repeat(60));
  console.log();
  console.log('📖 Learn more:');
  console.log('   - Documentation: https://aethermind.ai/documentation');
  console.log('   - GitHub: https://github.com/United-Visions/AetherAGI');
  console.log('   - Discord: https://discord.gg/aethermind');
  console.log();
  console.log('🚀 Ready to build with real AGI? Get your API key at:');
  console.log('   https://aethermind.ai');
  console.log();
}

// Run the demo
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
