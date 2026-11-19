#import <Cocoa/Cocoa.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int startupError(NSString *detail, BOOL worker) {
    fprintf(stderr, "%s\n", detail.UTF8String);
    if (!worker) {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        NSAlert *alert = [NSAlert new];
        alert.messageText = @"PURE RGB 无法启动";
        alert.informativeText = detail;
        [alert runModal];
    }
    return 1;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        BOOL worker = argc >= 3 && strcmp(argv[1], "-m") == 0 &&
            (strcmp(argv[2], "pure_rgb.workflow_worker") == 0 ||
             strcmp(argv[2], "pure_rgb.live_worker") == 0);
        NSBundle *bundle = NSBundle.mainBundle;
        NSString *resources = bundle.resourcePath;
        NSDictionary *config = bundle.infoDictionary;
        NSString *workspace = config[@"PURE RGB Test Workspace"];
        NSString *deps = config[@"PURE RGB Python Dependencies"];
        NSString *source = [resources stringByAppendingPathComponent:@"Source"];
        NSString *runtime = [resources stringByAppendingPathComponent:@"Python"];
        NSString *libraryName = config[@"PURE RGB Python Library"];
        NSFileManager *files = NSFileManager.defaultManager;
        if (!workspace.length || !deps.length || !resources.length || !libraryName.length ||
            ![files isReadableFileAtPath:[workspace stringByAppendingPathComponent:@"vendor/negpy/negpy/__init__.py"]] ||
            ![files isReadableFileAtPath:[deps stringByAppendingPathComponent:@"PyQt6"]] ||
            chdir(workspace.fileSystemRepresentation) != 0) {
            return startupError(@"此应用使用构建时的 PURE RGB 源码目录及其运行依赖。请恢复该目录的位置，或在新位置重新构建。", worker);
        }
        NSString *packages = [NSString stringWithFormat:@"%@:%@", source, deps];
        setenv("PURE_RGB_TEST_WORKSPACE", workspace.fileSystemRepresentation, 1);
        setenv("PYTHONHOME", runtime.fileSystemRepresentation, 1);
        setenv("PYTHONPATH", packages.fileSystemRepresentation, 1);
        setenv("PYTHONSAFEPATH", "1", 1);
        setenv("PYTHONNOUSERSITE", "1", 1);
        setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
        if (!getenv("PATH")) {
            setenv("PATH", "/usr/bin:/bin:/usr/sbin:/sbin", 1);
        }
        NSString *library = [runtime stringByAppendingPathComponent:libraryName];
        void *python = dlopen(library.fileSystemRepresentation, RTLD_NOW | RTLD_GLOBAL);
        if (!python) {
            return startupError([NSString stringWithFormat:@"无法加载打包的 Python 运行时。\n%s", dlerror()], worker);
        }
        int (*pythonMain)(int, char **) = dlsym(python, "Py_BytesMain");
        if (!pythonMain) {
            return startupError(@"打包的 Python 运行时缺少启动入口。", worker);
        }
        NSString *executable __attribute__((objc_precise_lifetime)) = bundle.executablePath;
        argv[0] = (char *)executable.fileSystemRepresentation;
        if (worker) {
            return pythonMain(argc, argv);
        }
        char **arguments = calloc((size_t)argc + 3, sizeof(char *));
        if (!arguments) {
            return startupError(@"启动时无法分配内存。", worker);
        }
        arguments[0] = argv[0];
        arguments[1] = "-m";
        arguments[2] = "pure_rgb.workflow_launcher";
        int argumentCount = 3;
        for (int i = 1; i < argc; ++i) {
            if (strncmp(argv[i], "-psn_", 5) != 0) {
                arguments[argumentCount++] = argv[i];
            }
        }
        int status = pythonMain(argumentCount, arguments);
        free(arguments);
        return status;
    }
}
