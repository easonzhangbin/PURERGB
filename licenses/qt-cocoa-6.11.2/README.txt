Qt accessibility fixes for PURE RGB binary releases

The bundled Qt runtime remains Qt 6.11.2. Its Cocoa platform plugin and QtWidgets
framework are rebuilt; the two changes and their corresponding source are below.
The change backports the two parent-ownership guards for QTBUG-149612. Synthesized
accessibility cells must not delete their parent table's interface. Accessibility
remains enabled.

Unmodified corresponding source (Qt 6.11.2, commit ef55f427f2c8b410d34f8a7681020a3000cf6866):
https://github.com/qt/qtbase/tree/ef55f427f2c8b410d34f8a7681020a3000cf6866
https://github.com/qt/qtbase/archive/ef55f427f2c8b410d34f8a7681020a3000cf6866.tar.gz

Upstream issue and fix:
https://qt-project.atlassian.net/browse/QTBUG-149612
https://github.com/qt/qtbase/commit/8148c48deb8e4345bacb797cb75c011df6f797fa

The exact local changes are in QTBUG-149612.patch. The complete modified
qcocoaaccessibilityelement.mm and the standalone CMake wrapper are included here.
The original source's copyright and license notices are retained. License texts
are included alongside this file; the plugin source permits GPL-3.0-only.

Build outline:
1. Obtain the source commit above and apply QTBUG-149612.patch.
2. Install the matching official Qt 6.11.2 macOS SDK's qtbase archive, including
   private headers and BuildInternals CMake files. For example, use aqtinstall:
   python -m aqt install-qt mac desktop 6.11.2 clang_64 -O SDK_DIR --archives qtbase
3. In CMakeLists.txt set the add_subdirectory path to the source's
   src/plugins/platforms/cocoa directory.
4. Configure with CMake/Ninja, CMAKE_PREFIX_PATH=SDK_DIR/6.11.2/macos,
   CMAKE_OSX_ARCHITECTURES=arm64, CMAKE_OSX_DEPLOYMENT_TARGET=14.0,
   CMAKE_BUILD_TYPE=Release, QT_SKIP_AUTO_PLUGIN_INCLUSION=ON and
   QT_GENERATE_SBOM=OFF. Build the QCocoaIntegrationPlugin target.
5. Replace the SDK absolute LC_RPATH in libqcocoa.dylib with
   @loader_path/../../lib, then sign the staged plugin and application.

The packaged PURE-Qt-runtime.json records the exact patch and plugin SHA-256.

Additional QtWidgets accessibility cache fix

A stale childToId entry can resolve to a null interface after native accessibility
cache cleanup. Qt Release builds compile out Q_ASSERT; the next role() dereferences
the null pointer during row insertion/removal. The bundled QtWidgets framework
skips these dead cache entries and preserves the remaining valid interfaces.
This is a local defensive backport, separate from the official Cocoa ownership fix.
Source: QtBase ef55f427f2c8b410d34f8a7681020a3000cf6866, src/widgets/accessible/itemviews.cpp.
See widgets-accessibility-cache.patch, itemviews.cpp and Widgets-CMakeLists.txt.
Build with the same 6.11.2 SDK/Core/Gui/private headers as the runtime, arm64,
CMAKE_OSX_DEPLOYMENT_TARGET=14.0. Before building Widgets, copy the SDK framework
Headers directory into build/include/QtWidgets to provide generated dependency headers.
Set QT_SKIP_AUTO_PLUGIN_INCLUSION=ON and QT_GENERATE_SBOM=OFF.
The installed framework must retain its @rpath/QtWidgets.framework/Versions/A/QtWidgets ID.
Use @loader_path/../../.. as its runtime library search path and re-sign locally.
