<?php
require_once('db.php');

if (!isset($_COOKIE['User'])) {
    header('Location: /login.php');
    exit();
}

$error = '';
$message = '';
$posts = [];

$link = mysqli_connect($servername, $username, $password, $dbName);

if (!$link) {
    die('Ошибка подключения: ' . mysqli_connect_error());
}

mysqli_set_charset($link, 'utf8mb4');

if (isset($_POST['submit_post'])) {
    $title = trim($_POST['postTitle'] ?? '');
    $main_text = trim($_POST['postContent'] ?? '');
    $imagePath = '';

    if (!$title || !$main_text) {
        $error = 'no data post';
    } else {
        if (!empty($_FILES['file']['name'])) {
            if (!isset($_FILES['file']['error']) || $_FILES['file']['error'] !== UPLOAD_ERR_OK) {
                $error = 'Ошибка загрузки файла';
            } else {
                $allowedExtensions = ['gif', 'jpg', 'jpeg', 'png', 'webp'];
                $fileExtension = strtolower(pathinfo($_FILES['file']['name'], PATHINFO_EXTENSION));

                if (!in_array($fileExtension, $allowedExtensions, true)) {
                    $error = 'Можно загружать только изображения gif, jpg, jpeg, png, webp';
                } elseif ($_FILES['file']['size'] > 5 * 1024 * 1024) {
                    $error = 'Файл слишком большой. Максимум 5 МБ';
                } else {
                    $uploadDir = __DIR__ . '/upload/';

                    if (!is_dir($uploadDir) && !mkdir($uploadDir, 0777, true)) {
                        $error = 'Не удалось создать папку upload';
                    } else {
                        $safeBaseName = preg_replace('/[^A-Za-z0-9._-]/', '_', basename($_FILES['file']['name']));
                        $fileName = uniqid('post_', true) . '_' . $safeBaseName;
                        $targetPath = $uploadDir . $fileName;

                        if (move_uploaded_file($_FILES['file']['tmp_name'], $targetPath)) {
                            $imagePath = 'upload/' . $fileName;
                        } else {
                            $error = 'Не удалось сохранить файл';
                        }
                    }
                }
            }
        }

        if ($error === '') {
            $usernameCookie = mysqli_real_escape_string($link, $_COOKIE['User']);
            $titleSql = mysqli_real_escape_string($link, $title);
            $mainTextSql = mysqli_real_escape_string($link, $main_text);

            if ($imagePath !== '') {
                $imageSql = mysqli_real_escape_string($link, $imagePath);
                $sql = "INSERT INTO posts (username, title, main_text, image_path) VALUES ('$usernameCookie', '$titleSql', '$mainTextSql', '$imageSql')";
            } else {
                $sql = "INSERT INTO posts (username, title, main_text, image_path) VALUES ('$usernameCookie', '$titleSql', '$mainTextSql', NULL)";
            }

            if (!mysqli_query($link, $sql)) {
                $error = 'error insert data post';
            } else {
                header('Location: /profile.php?created=1');
                mysqli_close($link);
                exit();
            }
        }
    }
}

if (isset($_GET['created'])) {
    $message = 'Пост успешно опубликован';
}

$usernameCookie = mysqli_real_escape_string($link, $_COOKIE['User']);
$sql = "SELECT * FROM posts WHERE username='$usernameCookie' ORDER BY id DESC";
$res = mysqli_query($link, $sql);

if ($res) {
    while ($post = mysqli_fetch_array($res)) {
        $posts[] = $post;
    }
} else {
    $error = 'Не удалось получить список постов';
}

mysqli_close($link);
?>
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Избосарова Ж.О. - профиль</title>

    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>

<body>

    <nav class="navbar navbar-expand-lg navbar-custom fixed-top">

        <div class="container">

            <a class="navbar-brand" href="/index.php">
                PONYGRAM
            </a>

            <div class="d-flex align-items-center gap-3">
                <span class="text-muted">
                    <?php echo htmlspecialchars($_COOKIE['User'], ENT_QUOTES, 'UTF-8'); ?>
                </span>

                <?php if (isset($_COOKIE['User'])): ?>
                    <form action="/logout.php" method="POST" class="d-flex">
                        <button class="btn btn-outline-danger rounded-pill px-4" type="submit">Logout</button>
                    </form>
                <?php endif; ?>
            </div>

        </div>

    </nav>

    <div class="feed-container">

        <div class="create-post-area">

            <h4 class="mb-3">
                Поделиться моментом ✨
            </h4>

            <?php if ($message !== ''): ?>
                <div class="alert alert-success py-2" role="alert">
                    <?php echo htmlspecialchars($message, ENT_QUOTES, 'UTF-8'); ?>
                </div>
            <?php endif; ?>

            <?php if ($error !== ''): ?>
                <div class="alert alert-danger py-2" role="alert">
                    <?php echo htmlspecialchars($error, ENT_QUOTES, 'UTF-8'); ?>
                </div>
            <?php endif; ?>

            <form action="/profile.php" method="POST" enctype="multipart/form-data">
                <input
                    type="text"
                    id="newPostTitle"
                    name="postTitle"
                    class="form-control"
                    placeholder="Название поста"
                    required
                >

                <textarea
                    id="newPostContent"
                    name="postContent"
                    class="form-control"
                    rows="4"
                    placeholder="Поделись своими мыслями..."
                    required
                ></textarea>

                <div class="d-flex justify-content-between align-items-center gap-3">

                    <input type="file" name="file" accept=".gif,.jpg,.jpeg,.png,.webp,image/*" class="form-control">

                    <button type="submit" name="submit_post" class="btn btn-pink px-4">
                        Опубликовать
                    </button>

                </div>
            </form>

        </div>

        <?php if (count($posts) > 0): ?>
            <?php foreach ($posts as $post): ?>
                <div class="post-card">

                    <?php if (!empty($post['image_path'])): ?>
                        <div class="post-img-container">
                            <img src="<?php echo htmlspecialchars($post['image_path'], ENT_QUOTES, 'UTF-8'); ?>" alt="<?php echo htmlspecialchars($post['title'], ENT_QUOTES, 'UTF-8'); ?>">
                        </div>
                    <?php endif; ?>

                    <h4><?php echo htmlspecialchars($post['title'], ENT_QUOTES, 'UTF-8'); ?></h4>

                    <p class="text-muted">
                        <?php echo nl2br(htmlspecialchars($post['main_text'], ENT_QUOTES, 'UTF-8')); ?>
                    </p>

                    <div class="d-flex justify-content-between align-items-center">

                        <span class="text-muted">Пост #<?php echo (int) $post['id']; ?></span>

                        <a href="/posts.php?id=<?php echo (int) $post['id']; ?>" class="magic-link">
                            Открыть →
                        </a>

                    </div>

                </div>
            <?php endforeach; ?>
        <?php else: ?>
            <div class="post-card">
                <h4>Пока нет публикаций</h4>
                <p class="text-muted mb-0">
                    Создай первый пост через форму выше, и он сразу появится в профиле и на главной странице.
                </p>
            </div>
        <?php endif; ?>

    </div>

    <script src="js/scripts.js"></script>

</body>
</html>
