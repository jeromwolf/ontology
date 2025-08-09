#!/usr/bin/env node

/**
 * Test script for KSS Content Management System
 * Run with: node scripts/test-content-manager.js
 */

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000'

// ANSI color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
}

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`)
}

async function testEndpoint(endpoint, method = 'GET', body = null) {
  try {
    log(`\nTesting: ${method} ${endpoint}`, 'cyan')
    
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
      }
    }
    
    if (body) {
      options.body = JSON.stringify(body)
    }
    
    const response = await fetch(`${BASE_URL}${endpoint}`, options)
    const data = await response.json()
    
    if (response.ok) {
      log(`✓ Success (${response.status})`, 'green')
      return { success: true, data }
    } else {
      log(`✗ Failed (${response.status})`, 'red')
      return { success: false, error: data }
    }
  } catch (error) {
    log(`✗ Error: ${error.message}`, 'red')
    return { success: false, error: error.message }
  }
}

async function runTests() {
  log('\n' + '='.repeat(50), 'bright')
  log('KSS Content Manager Test Suite', 'bright')
  log('='.repeat(50) + '\n', 'bright')
  
  let passedTests = 0
  let failedTests = 0
  
  // Test 1: Get all module statuses
  log('Test 1: Fetching module statuses...', 'yellow')
  const modulesResult = await testEndpoint('/api/content-manager/modules')
  if (modulesResult.success) {
    log(`Found ${modulesResult.data.length} modules`, 'blue')
    passedTests++
  } else {
    failedTests++
  }
  
  // Test 2: Get validation issues
  log('\nTest 2: Fetching validation issues...', 'yellow')
  const issuesResult = await testEndpoint('/api/content-manager/validation/issues')
  if (issuesResult.success) {
    log(`Found ${issuesResult.data.length} validation issues`, 'blue')
    
    // Display issue summary
    const severityCounts = {}
    issuesResult.data.forEach(issue => {
      severityCounts[issue.severity] = (severityCounts[issue.severity] || 0) + 1
    })
    
    Object.entries(severityCounts).forEach(([severity, count]) => {
      const color = severity === 'critical' ? 'red' : 
                    severity === 'high' ? 'yellow' : 'cyan'
      log(`  - ${severity}: ${count} issues`, color)
    })
    passedTests++
  } else {
    failedTests++
  }
  
  // Test 3: Get content updates
  log('\nTest 3: Fetching content updates...', 'yellow')
  const updatesResult = await testEndpoint('/api/content-manager/updates')
  if (updatesResult.success) {
    log(`Found ${updatesResult.data.length} content updates`, 'blue')
    
    // Display update summary
    const statusCounts = {}
    updatesResult.data.forEach(update => {
      statusCounts[update.status] = (statusCounts[update.status] || 0) + 1
    })
    
    Object.entries(statusCounts).forEach(([status, count]) => {
      log(`  - ${status}: ${count} updates`, 'cyan')
    })
    passedTests++
  } else {
    failedTests++
  }
  
  // Test 4: Run validation for a specific module
  log('\nTest 4: Running validation for LLM module...', 'yellow')
  const validationResult = await testEndpoint(
    '/api/content-manager/validation/run?module=llm',
    'POST'
  )
  if (validationResult.success) {
    log(`Validation completed, found ${validationResult.data.length} issues`, 'blue')
    passedTests++
  } else {
    failedTests++
  }
  
  // Test 5: Check for updates (specific module)
  log('\nTest 5: Checking for updates in Stock Analysis module...', 'yellow')
  const checkResult = await testEndpoint(
    '/api/content-manager/updates/check?module=stock-analysis',
    'POST'
  )
  if (checkResult.success) {
    log(`Update check completed, found ${checkResult.data.length} potential updates`, 'blue')
    passedTests++
  } else {
    failedTests++
  }
  
  // Test 6: Daily check endpoint (GET info)
  log('\nTest 6: Checking daily cron endpoint...', 'yellow')
  const cronResult = await testEndpoint('/api/content-manager/cron/daily-check')
  if (cronResult.success) {
    log(`Cron endpoint is configured`, 'blue')
    passedTests++
  } else {
    failedTests++
  }
  
  // Summary
  log('\n' + '='.repeat(50), 'bright')
  log('Test Summary', 'bright')
  log('='.repeat(50), 'bright')
  
  const totalTests = passedTests + failedTests
  const passRate = ((passedTests / totalTests) * 100).toFixed(1)
  
  log(`\nTotal Tests: ${totalTests}`, 'cyan')
  log(`Passed: ${passedTests}`, 'green')
  log(`Failed: ${failedTests}`, failedTests > 0 ? 'red' : 'green')
  log(`Pass Rate: ${passRate}%`, passRate >= 80 ? 'green' : 'yellow')
  
  // Module Health Summary
  if (modulesResult.success && modulesResult.data.length > 0) {
    log('\n' + '='.repeat(50), 'bright')
    log('Module Health Summary', 'bright')
    log('='.repeat(50), 'bright')
    
    modulesResult.data.forEach(module => {
      const healthColor = module.simulatorHealth === 'healthy' ? 'green' :
                         module.simulatorHealth === 'warning' ? 'yellow' : 'red'
      
      log(`\n${module.name}:`, 'cyan')
      log(`  Health: ${module.simulatorHealth}`, healthColor)
      log(`  Accuracy: ${module.accuracyScore.toFixed(1)}%`, 
          module.accuracyScore >= 90 ? 'green' : 
          module.accuracyScore >= 70 ? 'yellow' : 'red')
      log(`  Issues: ${module.outdatedChapters} outdated, ${module.brokenLinks} broken links, ${module.deprecatedCode} deprecated`)
      log(`  Last Update: ${new Date(module.lastUpdate).toLocaleDateString()}`)
    })
  }
  
  log('\n')
  process.exit(failedTests > 0 ? 1 : 0)
}

// Check if server is running
async function checkServer() {
  try {
    const response = await fetch(BASE_URL)
    if (response.ok) {
      return true
    }
  } catch (error) {
    return false
  }
  return false
}

// Main execution
async function main() {
  log(`Testing Content Manager at: ${BASE_URL}`, 'cyan')
  
  const serverRunning = await checkServer()
  if (!serverRunning) {
    log('\n⚠️  Server is not running!', 'red')
    log('Please start the development server with: npm run dev', 'yellow')
    process.exit(1)
  }
  
  await runTests()
}

main().catch(error => {
  log(`\nFatal error: ${error.message}`, 'red')
  process.exit(1)
})