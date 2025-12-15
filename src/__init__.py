from .model import (
    TeacherGCN,
    StudentModel,
    SGC,
    SSGC,
    criterion_inter_layer,
    criterion_distill,
)

from .main_distiller import (
    main,
    main_distill,
    train_distill,
    test_distill,
    train_teacher,
)

__all__ = [
    "TeacherGCN",
    "StudentModel",
    "SGC",
    "SSGC",
    "criterion_inter_layer",
    "criterion_distill",
    "main",
    "main_distill",
    "train_distill",
    "test_distill",
    "train_teacher",
]
