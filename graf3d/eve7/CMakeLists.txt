# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTEve
  HEADERS
    ROOT/REveChunkManager.hxx
    ROOT/REveClient.hxx
    ROOT/REveCompound.hxx
    ROOT/REveBox.hxx
    ROOT/REveBoxSet.hxx
    ROOT/REveCaloData.hxx
    ROOT/REveCalo.hxx
    ROOT/REveDataCollection.hxx
    ROOT/REveDataProxyBuilderBase.hxx
    ROOT/REveDataSimpleProxyBuilder.hxx
    ROOT/REveDataSimpleProxyBuilderTemplate.hxx
    ROOT/REveDataTable.hxx
    ROOT/REveDigitSet.hxx
    ROOT/REveElement.hxx
    ROOT/REveFrameBox.hxx
    ROOT/REveGeoPolyShape.hxx
    ROOT/REveGeoShapeExtract.hxx
    ROOT/REveGeoShape.hxx
    ROOT/REveGeoTopNode.hxx
    ROOT/REveGluTess.hxx
    ROOT/REveJetCone.hxx
    ROOT/REveEllipsoid.hxx
    ROOT/REveLine.hxx
    ROOT/REveManager.hxx
    ROOT/REvePathMark.hxx
    ROOT/REvePointSet.hxx
    ROOT/REvePolygonSetProjected.hxx
    ROOT/REveProjectionBases.hxx
    ROOT/REveProjectionManager.hxx
    ROOT/REveProjections.hxx
    ROOT/REveRenderData.hxx
    ROOT/REveRGBAPalette.hxx
    ROOT/REveScalableStraightLineSet.hxx
    ROOT/REveScene.hxx
    ROOT/REveSceneInfo.hxx
    ROOT/REveSecondarySelectable.hxx
    ROOT/REveSelection.hxx
    ROOT/REveShape.hxx
    ROOT/REveStraightLineSet.hxx
    ROOT/REveSystem.hxx
    ROOT/REveTableInfo.hxx
    ROOT/REveTableProxyBuilder.hxx
    ROOT/REveText.hxx
    ROOT/REveTrack.hxx
    ROOT/REveTrackProjected.hxx
    ROOT/REveTrackPropagator.hxx
    ROOT/REveTrans.hxx
    ROOT/REveTreeTools.hxx
    ROOT/REveTypes.hxx
    ROOT/REveUtil.hxx
    ROOT/REveVector.hxx
    ROOT/REveViewer.hxx
    ROOT/REveViewContext.hxx
    ROOT/REveVSD.hxx
    ROOT/REveVSDStructs.hxx
  SOURCES
    glu/memalloc.c
    glu/dict.c
    glu/geom.c
    glu/mesh.c
    glu/priorityq.c
    glu/normal.c
    glu/render.c
    glu/sweep.c
    glu/tess.c
    glu/tessmono.c
    src/REveChunkManager.cxx
    src/REveBox.cxx
    src/REveBoxSet.cxx
    src/REveCalo.cxx
    src/REveCaloData.cxx
    src/REveCompound.cxx
    src/REveDataCollection.cxx
    src/REveDataProxyBuilderBase.cxx
    src/REveDataSimpleProxyBuilder.cxx
    src/REveDataTable.cxx
    src/REveDigitSet.cxx
    src/REveFrameBox.cxx
    src/REveElement.cxx
    src/REveGeoPolyShape.cxx
    src/REveGeoShape.cxx
    src/REveGeoShapeExtract.cxx
    src/REveGeoTopNode.cxx
    src/REveGluTess.cxx
    src/REveJetCone.cxx
    src/REveEllipsoid.cxx
    src/REveLine.cxx
    src/REveManager.cxx
    src/REvePathMark.cxx
    src/REvePointSet.cxx
    src/REvePolygonSetProjected.cxx
    src/REveProjectionBases.cxx
    src/REveProjectionManager.cxx
    src/REveProjections.cxx
    src/REveRenderData.cxx
    src/REveRGBAPalette.cxx
    src/REveScalableStraightLineSet.cxx
    src/REveScene.cxx
    src/REveSceneInfo.cxx
    src/REveSecondarySelectable.cxx
    src/REveSelection.cxx
    src/REveShape.cxx
    src/REveStraightLineSet.cxx
    src/REveTableInfo.cxx
    src/REveTableProxyBuilder.cxx
    src/REveText.cxx
    src/REveTrack.cxx
    src/REveTrackProjected.cxx
    src/REveTrackPropagator.cxx
    src/REveTrans.cxx
    src/REveTreeTools.cxx
    src/REveTypes.cxx
    src/REveUtil.cxx
    src/REveVector.cxx
    src/REveViewer.cxx
    src/REveVSD.cxx
    src/REveVSDStructs.cxx
  DEPENDENCIES
    Core
    Geom
    Physics
    EG
    TreePlayer
    RCsg
    ROOTWebDisplay
    ROOTBrowserv7
    ROOTGeomViewer
  ${EXTRA_DICT_OPTS}
)

target_link_libraries(ROOTEve PUBLIC nlohmann_json::nlohmann_json)
# this is required for glew
target_include_directories(ROOTEve PRIVATE ${CMAKE_SOURCE_DIR}/graf3d/eve7)

ROOT_ADD_TEST_SUBDIRECTORY(test)
