'use client';

import { useState } from 'react';
import {
  GitBranch, Plus, Play, Calendar,
  Clock, Mail, AlertCircle, Download
} from 'lucide-react';

interface Task {
  id: string;
  name: string;
  operator: string;
  dependencies: string[];
}

export default function AirflowDAGBuilder() {
  const [dagName, setDagName] = useState('my_data_pipeline');
  const [schedule, setSchedule] = useState('0 2 * * *');
  const [tasks, setTasks] = useState<Task[]>([
    { id: 'extract', name: 'Extract Data', operator: 'PythonOperator', dependencies: [] },
  ]);
  const [selectedTask, setSelectedTask] = useState<string | null>(null);

  const operators = [
    'PythonOperator',
    'BashOperator',
    'PostgresOperator',
    'SnowflakeOperator',
    'SparkSubmitOperator',
    'EmailOperator',
  ];

  const schedulePresets = [
    { label: '매일 오전 2시', value: '0 2 * * *' },
    { label: '매시간', value: '0 * * * *' },
    { label: '매주 월요일', value: '0 0 * * 1' },
    { label: '매월 1일', value: '0 0 1 * *' },
  ];

  const addTask = () => {
    const newTask: Task = {
      id: `task_${tasks.length + 1}`,
      name: `Task ${tasks.length + 1}`,
      operator: 'PythonOperator',
      dependencies: [],
    };
    setTasks([...tasks, newTask]);
  };

  const generateCode = () => {
    return `from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    '${dagName}',
    default_args=default_args,
    description='Generated data pipeline',
    schedule_interval='${schedule}',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['auto-generated'],
) as dag:
${tasks.map(task => `
    ${task.id} = ${task.operator}(
        task_id='${task.id}',
        python_callable=${task.id}_function,  # 실제 함수 정의 필요
    )`).join('')}

    # Dependencies
${tasks.filter(t => t.dependencies.length > 0).map(task =>
  `    ${task.dependencies.join(' >> ')} >> ${task.id}`
).join('\n')}
`;
  };

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl p-6 text-white">
        <div className="flex items-center gap-3 mb-2">
          <GitBranch size={32} />
          <h2 className="text-2xl font-bold">Airflow DAG 빌더</h2>
        </div>
        <p className="text-cyan-100">시각적으로 Airflow DAG를 설계하고 Python 코드를 생성하세요</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4">⚙️ DAG 설정</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-semibold mb-2">DAG 이름</label>
              <input
                type="text"
                value={dagName}
                onChange={(e) => setDagName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
              />
            </div>
            <div>
              <label className="block text-sm font-semibold mb-2">스케줄 (Cron)</label>
              <select
                value={schedule}
                onChange={(e) => setSchedule(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
              >
                {schedulePresets.map((preset) => (
                  <option key={preset.value} value={preset.value}>
                    {preset.label} ({preset.value})
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
          <h3 className="text-lg font-bold mb-4">📦 Task 추가</h3>
          <button
            onClick={addTask}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-500 hover:bg-blue-600 text-white rounded-lg font-semibold"
          >
            <Plus size={20} /> Add Task
          </button>
          <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
            총 {tasks.length}개 Task
          </div>
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">🎨 DAG 시각화</h3>
        <div className="space-y-3">
          {tasks.map((task, idx) => (
            <div key={task.id} className="flex items-center gap-4">
              <div className="flex-1 p-4 bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-300 rounded-lg">
                <div className="font-semibold">{task.name}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">{task.operator}</div>
              </div>
              {idx < tasks.length - 1 && (
                <div className="text-gray-400">→</div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-lg">
        <h3 className="text-lg font-bold mb-4">💻 생성된 Python 코드</h3>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto max-h-96">
          {generateCode()}
        </pre>
      </div>
    </div>
  );
}
