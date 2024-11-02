from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime
from airflow import configuration as conf
from airflow.operators.email import EmailOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
# Enable pickle support for XCom
conf.set('core', 'enable_xcom_pickling', 'True')
default_args = {
    'owner': 'MLOPs_Team10',
    'depends_on_past': False,
    'start_date': datetime(2022, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1
}
dag = DAG(
    'dummy_operator_example',
    default_args=default_args,
    description='Dag example for Lab 1 of Airflow series',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
    )
start_task = DummyOperator(
    task_id='start_task',
    dag=dag
)
task_1 = DummyOperator(
    task_id='task_1',
    dag=dag
)
task_2 = DummyOperator(
    task_id='task_2',
    dag=dag
    )
end_task = DummyOperator(
    task_id='end_task',
    dag=dag
)
start_task >> task_1 >> task_2 >> end_task
