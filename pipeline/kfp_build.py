# pipelines/kfp_build.py
from kfp import dsl
from kfp.compiler import Compiler

@dsl.component
def echo_step(msg: str):
    print(msg)

@dsl.pipeline(name="face-enroll-recognize")
def face_enroll_recognize():
    echo_step(msg="enroll-batch started")
    echo_step(msg="recognize started")

if __name__ == "__main__":
    Compiler().compile(face_enroll_recognize, package_path="pipelines/face_enroll_recognize.json")
