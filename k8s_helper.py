import yaml
from kubernetes import client, config
from os import path
import json
import datetime

NAMESPACE = "second-carrier-prediction"
IMAGE = "registry.ailab.rnd.ki.sw.ericsson.se/second-carrier-prediction/main/fl-moe"
num_trainers = 8
def default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

def create_job_object(command, gen_name="eisamar-fed-moe-job-"):

    resources = client.V1ResourceRequirements(
        limits={"cpu": 46, "memory": f"{8*num_trainers}Gi", "nvidia.com/gpu": "8"},
        requests={"cpu": 46, "memory": f"{8*num_trainers}Gi", "nvidia.com/gpu": "8"})

    volumeMounts = client.V1VolumeMount(
        name="projectdisk",
        mount_path="/proj/second-carrier-prediction/")

    container = client.V1Container(
        name="fed-moe",
        image=IMAGE,
        command=command,
        volume_mounts=[volumeMounts],
        resources=resources,
        working_dir="/proj/second-carrier-prediction/federated-learning-mixture/")

    secrets = client.V1SecretReference(name="eisamar-fl-moe-token")

    #node_selector = {"nvidia.com/gpu": "true"}

    tolerations = client.V1Toleration(
        key="nvidia.com/gpu",
        operator="Exists",
        effect="NoSchedule")

    pvc = client.V1PersistentVolumeClaimVolumeSource(
        claim_name="cephfs-second-carrier-prediction")

    volumes = client.V1Volume(name="projectdisk", persistent_volume_claim=pvc)

    template = client.V1PodTemplateSpec(

        metadata=client.V1ObjectMeta(
            labels={"app": "fed-moe",
                    "ailab-job-type": "batch"}),

        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            tolerations=[tolerations],
            image_pull_secrets=[secrets],
            volumes=[volumes]
        ))

    spec = client.V1JobSpec(
        template=template,
        backoff_limit=4,
        active_deadline_seconds=3*24*60*60)

    metadata = client.V1ObjectMeta(
        generate_name=gen_name)

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=metadata,
        spec=spec)

    return job


def create_job(api_instance, job):
    api_response = api_instance.create_namespaced_job(
        body=job,
        namespace=NAMESPACE)
    return api_response.to_dict()


def main():
    config.load_kube_config()
    batch_v1 = client.BatchV1Api()
    job = create_job_object(
        ["python", "iterator_clusters.py", "-h"],
        gen_name="test-")
    return create_job(batch_v1, job)


if __name__ == '__main__':
    print(json.dumps(main(), default=default))
