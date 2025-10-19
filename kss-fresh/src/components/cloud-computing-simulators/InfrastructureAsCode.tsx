'use client';

import React, { useState } from 'react';
import { Code, Play, CheckCircle, XCircle, FileCode, Download } from 'lucide-react';

interface TerraformResource {
  type: string;
  name: string;
  properties: Record<string, any>;
}

export default function InfrastructureAsCode() {
  const [selectedTool, setSelectedTool] = useState<'terraform' | 'cloudformation' | 'pulumi'>('terraform');
  const [code, setCode] = useState(`# Terraform AWS Infrastructure

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"

  tags = {
    Name = "main-vpc"
  }
}

# Subnet
resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"

  tags = {
    Name = "public-subnet"
  }
}

# EC2 Instance
resource "aws_instance" "web" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.micro"
  subnet_id     = aws_subnet.public.id

  tags = {
    Name = "web-server"
  }
}

# S3 Bucket
resource "aws_s3_bucket" "data" {
  bucket = "my-app-data-bucket"

  tags = {
    Name = "data-bucket"
  }
}

# RDS Instance
resource "aws_db_instance" "database" {
  allocated_storage    = 20
  storage_type         = "gp2"
  engine               = "postgres"
  engine_version       = "15.3"
  instance_class       = "db.t3.micro"
  db_name              = "myapp"
  username             = "admin"
  password             = "changeme123"
  skip_final_snapshot  = true

  tags = {
    Name = "app-database"
  }
}`);

  const [deploymentStatus, setDeploymentStatus] = useState<'idle' | 'planning' | 'applying' | 'success' | 'error'>('idle');
  const [deploymentLogs, setDeploymentLogs] = useState<string[]>([]);
  const [resources, setResources] = useState<TerraformResource[]>([]);

  const templates = {
    terraform: {
      'basic-vpc': `# Basic VPC Configuration
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name = "main-vpc"
  }
}`,
      'lambda-function': `# Lambda Function
resource "aws_lambda_function" "api" {
  filename      = "lambda_function.zip"
  function_name = "api_handler"
  role          = aws_iam_role.lambda_role.arn
  handler       = "index.handler"
  runtime       = "nodejs18.x"

  environment {
    variables = {
      ENV = "production"
    }
  }
}`,
      'three-tier': `# Three-Tier Architecture
resource "aws_lb" "main" {
  name               = "app-lb"
  internal           = false
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
}

resource "aws_autoscaling_group" "web" {
  desired_capacity = 3
  max_size         = 5
  min_size         = 2
  vpc_zone_identifier = aws_subnet.private[*].id
}

resource "aws_db_instance" "main" {
  engine         = "postgres"
  instance_class = "db.t3.medium"
  multi_az       = true
}`
    },
    cloudformation: {
      'basic-vpc': `AWSTemplateFormatVersion: '2010-09-09'
Resources:
  MainVPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      Tags:
        - Key: Name
          Value: main-vpc`,
      'lambda-function': `Resources:
  ApiFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: api-handler
      Runtime: nodejs18.x
      Handler: index.handler
      Code:
        ZipFile: |
          exports.handler = async (event) => {
            return { statusCode: 200, body: 'Hello' };
          };`,
      'three-tier': `Resources:
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Type: application
      Subnets: !Ref PublicSubnets

  AutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      MinSize: 2
      MaxSize: 5
      DesiredCapacity: 3`
    }
  };

  const deploy = async () => {
    setDeploymentStatus('planning');
    setDeploymentLogs(['Starting deployment...', 'Initializing Terraform...']);

    setTimeout(() => {
      setDeploymentLogs(prev => [...prev, 'Terraform initialized successfully']);
      setDeploymentLogs(prev => [...prev, 'Refreshing state...']);
      setDeploymentLogs(prev => [...prev, 'Planning infrastructure changes...']);
    }, 500);

    setTimeout(() => {
      setDeploymentStatus('applying');

      // Parse resources from code (simplified)
      const resourceMatches = code.matchAll(/resource\s+"([^"]+)"\s+"([^"]+)"/g);
      const parsedResources: TerraformResource[] = [];

      for (const match of resourceMatches) {
        parsedResources.push({
          type: match[1],
          name: match[2],
          properties: {}
        });
      }

      setResources(parsedResources);

      parsedResources.forEach((resource, idx) => {
        setTimeout(() => {
          setDeploymentLogs(prev => [...prev, `Creating ${resource.type}.${resource.name}...`]);
        }, 1000 + idx * 500);
      });
    }, 1500);

    setTimeout(() => {
      setDeploymentStatus('success');
      setDeploymentLogs(prev => [...prev, '', 'Apply complete! Resources created successfully.', `Total resources: ${resources.length}`]);
    }, 3000 + resources.length * 500);
  };

  const loadTemplate = (template: string) => {
    if (selectedTool === 'terraform') {
      setCode(templates.terraform[template as keyof typeof templates.terraform] || '');
    } else if (selectedTool === 'cloudformation') {
      setCode(templates.cloudformation[template as keyof typeof templates.cloudformation] || '');
    }
  };

  const exportCode = () => {
    const extension = selectedTool === 'terraform' ? 'tf' : selectedTool === 'cloudformation' ? 'yaml' : 'ts';
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `infrastructure.${extension}`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-2">
                Infrastructure as Code 실습 환경
              </h1>
              <p className="text-gray-600 dark:text-gray-300">
                Terraform, CloudFormation, Pulumi로 인프라를 코드로 관리하세요
              </p>
            </div>

            <div className="flex items-center gap-3">
              <select
                value={selectedTool}
                onChange={(e) => setSelectedTool(e.target.value as any)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="terraform">Terraform</option>
                <option value="cloudformation">CloudFormation</option>
                <option value="pulumi">Pulumi (Coming Soon)</option>
              </select>

              <button
                onClick={exportCode}
                className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg flex items-center gap-2 transition-colors"
              >
                <Download className="w-4 h-4" />
                Export
              </button>

              <button
                onClick={deploy}
                disabled={deploymentStatus === 'planning' || deploymentStatus === 'applying'}
                className="px-6 py-2 bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg font-semibold flex items-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play className="w-5 h-5" />
                Deploy
              </button>
            </div>
          </div>

          {/* Templates */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => loadTemplate('basic-vpc')}
              className="px-3 py-1.5 text-sm bg-blue-100 hover:bg-blue-200 dark:bg-blue-900/30 dark:hover:bg-blue-900/50 text-blue-700 dark:text-blue-300 rounded-lg transition-colors"
            >
              Basic VPC
            </button>
            <button
              onClick={() => loadTemplate('lambda-function')}
              className="px-3 py-1.5 text-sm bg-green-100 hover:bg-green-200 dark:bg-green-900/30 dark:hover:bg-green-900/50 text-green-700 dark:text-green-300 rounded-lg transition-colors"
            >
              Lambda Function
            </button>
            <button
              onClick={() => loadTemplate('three-tier')}
              className="px-3 py-1.5 text-sm bg-purple-100 hover:bg-purple-200 dark:bg-purple-900/30 dark:hover:bg-purple-900/50 text-purple-700 dark:text-purple-300 rounded-lg transition-colors"
            >
              Three-Tier Architecture
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Code Editor */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                  <FileCode className="w-5 h-5 text-indigo-500" />
                  Code Editor
                </h3>

                <span className="px-3 py-1 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300 rounded-lg text-sm font-semibold">
                  {selectedTool === 'terraform' ? '.tf' : selectedTool === 'cloudformation' ? '.yaml' : '.ts'}
                </span>
              </div>

              <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="w-full h-[600px] px-4 py-3 font-mono text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                spellCheck="false"
              />
            </div>

            {/* Deployment Logs */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">Deployment Logs</h3>

              <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
                {deploymentLogs.length === 0 ? (
                  <div className="text-gray-500 text-center py-8">
                    Click "Deploy" to start the deployment process
                  </div>
                ) : (
                  deploymentLogs.map((log, idx) => (
                    <div key={idx} className="mb-1 text-green-400">
                      {log && <span className="text-gray-500 mr-2">$</span>}
                      {log}
                    </div>
                  ))
                )}
              </div>

              {deploymentStatus !== 'idle' && (
                <div className="mt-4 flex items-center gap-2">
                  {deploymentStatus === 'planning' && (
                    <span className="text-yellow-600 dark:text-yellow-400 flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-yellow-600 border-t-transparent rounded-full animate-spin" />
                      Planning...
                    </span>
                  )}
                  {deploymentStatus === 'applying' && (
                    <span className="text-blue-600 dark:text-blue-400 flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                      Applying changes...
                    </span>
                  )}
                  {deploymentStatus === 'success' && (
                    <span className="text-green-600 dark:text-green-400 flex items-center gap-2">
                      <CheckCircle className="w-5 h-5" />
                      Deployment successful!
                    </span>
                  )}
                  {deploymentStatus === 'error' && (
                    <span className="text-red-600 dark:text-red-400 flex items-center gap-2">
                      <XCircle className="w-5 h-5" />
                      Deployment failed
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Resources Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sticky top-6">
              <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-4">Resources</h3>

              {resources.length === 0 ? (
                <div className="text-gray-500 text-center py-8 text-sm">
                  No resources deployed yet
                </div>
              ) : (
                <div className="space-y-3">
                  {resources.map((resource, idx) => (
                    <div key={idx} className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                      <div className="flex items-center gap-2 mb-1">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                        <span className="font-semibold text-sm text-gray-900 dark:text-gray-100">
                          {resource.type}
                        </span>
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400 ml-6">
                        {resource.name}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">IaC Best Practices</h4>

                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <span>Use version control (Git)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <span>Store state remotely (S3 + DynamoDB)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <span>Use modules for reusability</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <span>Implement CI/CD pipelines</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <span>Use workspaces for environments</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-500 mt-0.5">✓</span>
                    <span>Never commit secrets</span>
                  </li>
                </ul>
              </div>

              <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">Common Commands</h4>

                <div className="space-y-2">
                  <div className="bg-gray-900 p-2 rounded font-mono text-xs text-green-400">
                    $ terraform init
                  </div>
                  <div className="bg-gray-900 p-2 rounded font-mono text-xs text-green-400">
                    $ terraform plan
                  </div>
                  <div className="bg-gray-900 p-2 rounded font-mono text-xs text-green-400">
                    $ terraform apply
                  </div>
                  <div className="bg-gray-900 p-2 rounded font-mono text-xs text-green-400">
                    $ terraform destroy
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
