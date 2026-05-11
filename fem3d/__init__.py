# -*- coding: utf-8 -*-
"""
fem3d — управляющий слой программного комплекса МКЭ для трёхмерной
теплопроводности. Связан с C++ ядром fem_core через ctypes (модуль
core_bridge).

Структура пакета:
    core_bridge — граница с ядром (12 функций C-API через ctypes);
    mesh        — пресеты, справочник материалов, импорт сеток;
    problem     — высокоуровневое описание задачи и оркестрация расчёта;
    postprocess — экспорт VTU/CSV/отчёт;
    verify      — верификационные задачи T1–T4.
"""

from .core_bridge import (
    BC_DIRICHLET, BC_NEUMANN, BC_NONE, BC_ROBIN,
    CoreBridge, CoreError,
    FACE_NAMES, FACE_X_MINUS, FACE_X_PLUS,
    FACE_Y_MINUS, FACE_Y_PLUS, FACE_Z_MINUS, FACE_Z_PLUS,
    SolverInfo,
    VOLSRC_BOX, VOLSRC_SPHERE,
)
from .mesh import (
    BoxPreset, Material, MeshInfo,
    MATERIALS, PRESETS,
    SUPPORTED_IMPORT_EXTENSIONS,
    compute_mesh_info, import_mesh_file, import_msh, import_stl,
    import_step, import_vtu, material_by_name,
)
from .problem import (
    BoundaryCondition, BoxGeometry, PointSource, Problem, VolumeSource,
    template_all_convection, template_bottom_heat_top_cool, template_reset,
)

__version__ = "1.1.0"

__all__ = [
    # core_bridge
    "BC_DIRICHLET", "BC_NEUMANN", "BC_NONE", "BC_ROBIN",
    "CoreBridge", "CoreError",
    "FACE_NAMES", "FACE_X_MINUS", "FACE_X_PLUS",
    "FACE_Y_MINUS", "FACE_Y_PLUS", "FACE_Z_MINUS", "FACE_Z_PLUS",
    "SolverInfo",
    "VOLSRC_BOX", "VOLSRC_SPHERE",
    # mesh
    "BoxPreset", "Material", "MeshInfo",
    "MATERIALS", "PRESETS",
    "SUPPORTED_IMPORT_EXTENSIONS",
    "compute_mesh_info", "import_mesh_file", "import_msh", "import_stl",
    "import_step", "import_vtu", "material_by_name",
    # problem
    "BoundaryCondition", "BoxGeometry", "PointSource", "Problem", "VolumeSource",
    "template_all_convection", "template_bottom_heat_top_cool", "template_reset",
]
